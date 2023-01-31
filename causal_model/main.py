import os, time, sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
import torch as th
from DDI_module.RelationModel import genRelationEmbd
from causal_model.models import MSGCN
from tools.utils import get_optimizer,evaluate_model
import dgl
import random
import scipy.sparse as sp
from causal_model.Dataset import loadData
import warnings
import torch.nn.functional as F
from causal_model.models import calc_disc
from causal_model.models import MultipleOptimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
warnings.filterwarnings('ignore')

#/data/compound_relations_syn&ant_1v1.xlsx

def evaluation(args, model, userMask, y_true, graph, user_features, item_features, item_side_feat, T_f):
    user_embed, item_embed = model.encoder(graph, user_features, item_features, item_side_feat)
    userMask = userMask.long().cuda()
    userEmbed = th.repeat_interleave(user_embed[userMask],len(item_features),dim=0)
    item_embed = item_embed.repeat(len(userMask),1)
    logits_f = model.decoder(userEmbed, item_embed, T_f[userMask])
    pred_matrix = logits_f.reshape(len(userMask),len(item_features))
    top_N_res,F1_score, prec, rec_at_k, ndcg = evaluate_model(pred_matrix.detach().cpu().numpy(), y_true, top_n=args.top_n)
    return top_N_res,F1_score, prec, rec_at_k, ndcg

def run(args, devices):
    dev_id = devices[0]
    th.cuda.set_device(dev_id)
    train_trainMat, valid_trainMat, test_trainMat, \
    train_userId, train_pos_itemId, train_neg_itemId, \
    train_mask, valid_mask, test_mask,\
    train_y, valid_y, test_y,\
    train_drug_num, valid_drug_num, test_drug_num, num_user, num_item,\
    drugID, user_features, item_features, labels_f, labels_cf, T_f, \
    train_T_f, train_T_cf, sample_nodepairs_f, sample_nodepairs_cf = loadData(
        args.datadir, args.feat_norm, args.test_ratio, args.valid_ratio,
        args.K, args.max_patient_neighbor, args.max_drug_neighbor, args.num_np)

    user_features = user_features.to(dev_id)
    item_features = item_features.to(dev_id)
    T_f = T_f.to(dev_id)
    train_T_f = train_T_f.to(dev_id)
    train_T_cf = train_T_cf.to(dev_id)
    pos_w_f = th.FloatTensor([args.neg_rate]).to(dev_id)
    userId = train_userId.long().cuda()
    pos_itemId = train_pos_itemId.long().cuda()
    neg_itemId = train_neg_itemId.long().cuda()
    labels_f = labels_f.to(dev_id)
    labels_cf = labels_cf.to(dev_id)
    pos_w_cf = (labels_cf.shape[0] - labels_cf.sum()) / labels_cf.sum() + 1e-6

    if args.model_name=='ddi':
        item_side_feat = genRelationEmbd(drugID, args.item_side_units, args.side_feat_type).to(dev_id)
    else:
        item_side_feat = None

    u_i_adj = (train_trainMat != 0) * 1
    i_u_adj = u_i_adj.T
    u_u_adj = sp.csr_matrix((num_user, num_user))
    i_i_adj = sp.csr_matrix((num_item, num_item))
    adj = sp.vstack([sp.hstack([u_u_adj, u_i_adj]), sp.hstack([i_u_adj, i_i_adj])]).tocsr()
    src, dst = adj.nonzero()

    train_g = dgl.graph(data=(src, dst),idtype=th.int32,num_nodes=adj.shape[0],device=dev_id)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    model = MSGCN(user_features.shape[1],item_features.shape[1],args.hide_units,args.item_side_units,
                   args.model_activation, args.layerNum, args.model_name, args.dec).cuda()
    # learning_rate = args.train_lr
    optimizer = get_optimizer(args.train_optimizer)(model.parameters(), lr=args.train_lr, weight_decay=args.l2reg)
    optims = MultipleOptimizer(args.lr_scheduler, optimizer)
    print("Loading network finished ...\n")

    best_valid_rec = -np.inf
    no_better_valid = 0
    best_epoch = -1
    print("Start training ...")
    for epoch in range(1, args.train_max_epoch):
        if epoch > 1:
            t0 = time.time()
        model.train()
        user_embed, item_embed, logits_f, logits_cf = model(train_g, user_features, item_features,
                                                             item_side_feat,
                                                             userId, pos_itemId, neg_itemId,
                                                             train_T_f, train_T_cf)

        learning_rate = optims.update_lr(args.train_lr)
        optims.zero_grad()
        loss_disc = calc_disc(args.disc_func, user_embed, item_embed,  sample_nodepairs_f, sample_nodepairs_cf)
        loss_f = F.binary_cross_entropy_with_logits(logits_f, labels_f, pos_weight=pos_w_f)
        loss_cf = F.binary_cross_entropy_with_logits(logits_cf, labels_cf, pos_weight=pos_w_cf)
        total_loss = loss_f + args.alpha * loss_cf + args.beta * loss_disc
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optims.step()
        print("Epoch {}, total loss: {:.8f}, loss_disc: {:.8f}, loss_f: {:.8f}, loss_cf: {:.8f}"
              .format(epoch, total_loss.item(), loss_disc.item(), loss_f.item(), loss_cf.item()))

        userEmbed = th.repeat_interleave(user_embed[train_mask], len(item_features), dim=0)
        item_embed = item_embed.repeat(len(train_mask), 1)
        logits_f = model.decoder(userEmbed, item_embed, T_f[train_mask])
        pred_matrix = logits_f.reshape(len(train_mask), len(item_features))
        train_top_N_res, F1_score, prec, rec, ndcg = evaluate_model(pred_matrix.detach().cpu().numpy(), train_y,
                                                                    top_n=args.top_n)
        print('Epoch {}, Train F1_score@k={:.4f}, prec@k={:.4f}, Recall@k={:.4f}, NDCG@k={:.4f}'.format(epoch, F1_score, prec,
        rec, ndcg))


        # if epoch > 1:
            # epoch_time = time.time() - t0
            # print("Epoch {} time {}".format(epoch, epoch_time))

        #valid and test
        if epoch % args.train_valid_interval == 0:
            u_i_adj = (valid_trainMat != 0) * 1
            i_u_adj = u_i_adj.T
            u_u_adj = sp.csr_matrix((num_user, num_user))
            i_i_adj = sp.csr_matrix((num_item, num_item))
            adj = sp.vstack([sp.hstack([u_u_adj, u_i_adj]), sp.hstack([i_u_adj, i_i_adj])]).tocsr()
            src, dst = adj.nonzero()
            valid_g = dgl.graph(data=(src, dst), idtype=th.int32, num_nodes=adj.shape[0], device=dev_id)
            valid_top_N_res, valid_F1_score, valid_prec, valid_rec, valid_ndcg = evaluation(args=args,
                                  model=model,
                                  userMask=valid_mask,
                                  y_true=valid_y,
                                  graph=valid_g,
                                  user_features=user_features,
                                  item_features=item_features,
                                  item_side_feat=item_side_feat,
                                  T_f = T_f)
            print('Val F1_score@k={:.4f}, prec@k={:.4f}, Recall@k={:.4f}, NDCG@k={:.4f}'
                  .format(valid_F1_score, valid_prec, valid_rec,valid_ndcg))
            if valid_rec > best_valid_rec:
                pretrained_params = parameters_to_vector(model.parameters())
                best_train_rec = rec
                best_train_prec = prec
                best_train_f1_socre = F1_score
                best_train_ndcg = ndcg
                u_i_adj = (test_trainMat != 0) * 1
                i_u_adj = u_i_adj.T
                u_u_adj = sp.csr_matrix((num_user, num_user))
                i_i_adj = sp.csr_matrix((num_item, num_item))
                adj = sp.vstack([sp.hstack([u_u_adj, u_i_adj]), sp.hstack([i_u_adj, i_i_adj])]).tocsr()
                src, dst = adj.nonzero()
                test_g = dgl.graph(data=(src, dst), idtype=th.int32, num_nodes=adj.shape[0], device=dev_id)
                best_valid_rec = valid_rec
                best_valid_prec = valid_prec
                best_valid_f1_socre= valid_F1_score
                best_valid_ndcg = valid_ndcg
                no_better_valid = 0
                best_epoch = epoch
                test_top_N_res, test_F1_score, test_prec,test_rec,test_ndcg = evaluation(args=args,
                                     model=model,
                                     userMask=test_mask,
                                     y_true=test_y,
                                     graph=test_g,
                                  user_features=user_features,
                                  item_features=item_features,
                                  item_side_feat=item_side_feat,
                                  T_f = T_f)
                best_test_rec = test_rec
                best_test_f1_score = test_F1_score
                best_test_prec = test_prec
                best_test_ndcg = test_ndcg
                best_test_res = test_top_N_res

                print('Test F1_score@k={:.4f}, prec@k={:.4f}, Recall@k={:.4f}, NDCG@k={:.4f}'
                      .format(test_F1_score, test_prec,test_rec,test_ndcg))
            else:
                no_better_valid += 1
                if no_better_valid > args.train_early_stopping_patience:
                    print("Early stopping threshold reached. Stop training.")
                    break
                # if no_better_valid > args.train_decay_patience:
                #     new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                #     if new_lr < learning_rate:
                #         print("\tChange the LR to %g" % new_lr)
                #         learning_rate = new_lr
                #         for p in optimizer.param_groups:
                #             p['lr'] = learning_rate
                #         no_better_valid = 0
                #         print("Change the LR to %g" % new_lr)

    print('Best epoch Idx={},'
          ' Best Valid Recall@k={:.4f}, Best Test Recall@k={:.4f},'
          ' Best Valid Prec@k={:.4f}, Best Test Prec@k={:.4f},'
          ' Best Valid F1@k={:.4f}, Best Test F1@k={:.4f} '
          'Best Valid NDCG@k={:.4f}, Best Test NDCG@k={:.4f}'.format(
        best_epoch, best_valid_rec, best_test_rec, best_valid_prec,
        best_test_prec, best_valid_f1_socre, best_test_f1_score, best_valid_ndcg, best_test_ndcg))

    # decoder fine-tuning
    if args.epochs_ft:
        learning_rate = args.lr_ft
        optim_ft = th.optim.Adam(model.decoder.parameters(),
                                    lr=learning_rate,
                                    weight_decay=args.l2reg)
        vector_to_parameters(pretrained_params, model.parameters())
        model.encoder.eval()
        with th.no_grad():
            u_i_adj = (train_trainMat != 0) * 1
            i_u_adj = u_i_adj.T
            u_u_adj = sp.csr_matrix((num_user, num_user))
            i_i_adj = sp.csr_matrix((num_item, num_item))
            adj = sp.vstack([sp.hstack([u_u_adj, u_i_adj]), sp.hstack([i_u_adj, i_i_adj])]).tocsr()
            src, dst = adj.nonzero()
            train_g = dgl.graph(data=(src, dst), idtype=th.int32, num_nodes=adj.shape[0], device=dev_id)
            user_embed, item_embed = model.encoder(train_g, user_features, item_features, item_side_feat)

        best_params = None
        no_better_valid = 0

        for epoch in range(args.epochs_ft):
            model.decoder.train()
            optim_ft.zero_grad()

            userEmbed = user_embed[userId]
            posEmbed = item_embed[pos_itemId]
            negEmbed = item_embed[neg_itemId]
            userEmbed = th.cat((userEmbed, userEmbed), dim=0)
            itemEmbed = th.cat((posEmbed, negEmbed), dim=0)
            logits_f = model.decoder(userEmbed, itemEmbed, train_T_f)
            loss = F.binary_cross_entropy_with_logits(logits_f, labels_f, pos_weight=pos_w_f)
            loss.backward()
            optim_ft.step()
            print("Epoch {}, finetune loss: {:.8f}".format(epoch, loss.item()))

            userEmbed = th.repeat_interleave(user_embed[train_mask], len(item_features), dim=0)
            itemEmbed = item_embed.repeat(len(train_mask), 1)
            logits_f = model.decoder(userEmbed, itemEmbed, T_f[train_mask])
            pred_matrix = logits_f.reshape(len(train_mask), len(item_features))
            train_top_N_res, F1_score, prec, rec, ndcg = evaluate_model(pred_matrix.detach().cpu().numpy(), train_y,
                                                                        top_n=args.top_n)
            print(' FT Train Epoch {}, F1_score@k={:.4f}, prec@k={:.4f}, Recall@k={:.4f}, NDCG@k={:.4f}'.format(epoch, F1_score, prec,
                                                                                                  rec, ndcg))



            model.decoder.eval()
            with th.no_grad():
                if epoch % args.train_valid_interval == 0:
                    u_i_adj = (valid_trainMat != 0) * 1
                    i_u_adj = u_i_adj.T
                    u_u_adj = sp.csr_matrix((num_user, num_user))
                    i_i_adj = sp.csr_matrix((num_item, num_item))
                    adj = sp.vstack([sp.hstack([u_u_adj, u_i_adj]), sp.hstack([i_u_adj, i_i_adj])]).tocsr()
                    src, dst = adj.nonzero()
                    valid_g = dgl.graph(data=(src, dst), idtype=th.int32, num_nodes=adj.shape[0], device=dev_id)
                    valid_top_N_res, valid_F1_score, valid_prec, valid_rec, valid_ndcg = evaluation(args=args,
                                                                                                    model=model,
                                                                                                    userMask=valid_mask,
                                                                                                    y_true=valid_y,
                                                                                                    graph=valid_g,
                                                                                                    user_features=user_features,
                                                                                                    item_features=item_features,
                                                                                                    item_side_feat=item_side_feat,
                                                                                                    T_f=T_f)
                    print('Val F1_score@k={:.4f}, prec@k={:.4f}, Recall@k={:.4f}, NDCG@k={:.4f}'
                          .format(valid_F1_score, valid_prec, valid_rec, valid_ndcg))
                    if valid_rec > best_valid_rec:
                        best_params = parameters_to_vector(model.parameters())
                        best_train_rec = rec
                        best_train_prec = prec
                        best_train_f1_socre = F1_score
                        best_train_ndcg = ndcg
                        u_i_adj = (test_trainMat != 0) * 1
                        i_u_adj = u_i_adj.T
                        u_u_adj = sp.csr_matrix((num_user, num_user))
                        i_i_adj = sp.csr_matrix((num_item, num_item))
                        adj = sp.vstack([sp.hstack([u_u_adj, u_i_adj]), sp.hstack([i_u_adj, i_i_adj])]).tocsr()
                        src, dst = adj.nonzero()
                        test_g = dgl.graph(data=(src, dst), idtype=th.int32, num_nodes=adj.shape[0], device=dev_id)
                        best_valid_rec = valid_rec
                        best_valid_prec = valid_prec
                        best_valid_f1_socre = valid_F1_score
                        best_valid_ndcg = valid_ndcg
                        no_better_valid = 0
                        best_epoch = epoch
                        test_top_N_res, test_F1_score, test_prec, test_rec, test_ndcg = evaluation(args=args,
                                                                                                   model=model,
                                                                                                   userMask=test_mask,
                                                                                                   y_true=test_y,
                                                                                                   graph=test_g,
                                                                                                   user_features=user_features,
                                                                                                   item_features=item_features,
                                                                                                   item_side_feat=item_side_feat,
                                                                                                   T_f=T_f)
                        best_test_rec = test_rec
                        best_test_f1_score = test_F1_score
                        best_test_prec = test_prec
                        best_test_ndcg = test_ndcg
                        best_test_res = test_top_N_res

                        print('Test F1_score@k={:.4f}, prec@k={:.4f}, Recall@k={:.4f}, NDCG@k={:.4f}'
                              .format(test_F1_score, test_prec, test_rec, test_ndcg))
                    else:
                        no_better_valid += 1
                        if no_better_valid > args.train_early_stopping_patience \
                                and learning_rate <= args.train_min_lr:
                            print("Early stopping threshold reached. Stop training.")
                            break
                        if no_better_valid > args.train_decay_patience:
                            new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                            if new_lr < learning_rate:
                                print("\tChange the LR to %g" % new_lr)
                                learning_rate = new_lr
                                for p in optim_ft.param_groups:
                                    p['lr'] = learning_rate
                                no_better_valid = 0
                                print("Change the LR to %g" % new_lr)


    print('Best epoch Idx={},'
          ' Best Valid Recall@k={:.4f}, Best Test Recall@k={:.4f},'
          ' Best Valid Prec@k={:.4f}, Best Test Prec@k={:.4f},'
          ' Best Valid F1@k={:.4f}, Best Test F1@k={:.4f} '
          'Best Valid NDCG@k={:.4f}, Best Test NDCG@k={:.4f}'.format(
          best_epoch,best_valid_rec, best_test_rec, best_valid_prec,
        best_test_prec, best_valid_f1_socre,best_test_f1_score, best_valid_ndcg, best_test_ndcg))

if __name__ == '__main__':
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='models')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--datadir', type=str, default="../data")
    parser.add_argument('--feat_norm', type=str, default='l2')
    parser.add_argument('--test_ratio', type=float, default=0.2)  # 0.2
    parser.add_argument('--valid_ratio', type=float, default=0.3)  # 0.3
    parser.add_argument('--max_patient_neighbor', type=int, default=100)
    parser.add_argument('--max_drug_neighbor', type=int, default=20)
    parser.add_argument('--K', type=int, default=15, help="num of clusters for Kmeans to generate treatment")
    parser.add_argument('--num_np', type=int, default=1000,
                        help='number of sampled node pairs when calculating disc loss')
    parser.add_argument('--neg_rate', type=int, default=1, help='rate of negative samples during training')
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--layerNum', type=int, default=2) #ours=2
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--item_side_units', type=int, default=64) #32
    parser.add_argument('--hide_units', type=int, default=64)
    parser.add_argument('--dec', type=str, default="hadamard")
    parser.add_argument('--side_feat_type', type=str, default='onehot')
    parser.add_argument('--model_name', type=str, default='ddi') #'lightgcn','ddi'
    parser.add_argument('--alpha', type=float, default=1, help='weight of cf loss')
    parser.add_argument('--beta', type=float, default=0, help='weight of discrepancy loss')
    parser.add_argument('--disc_func', type=str, default='lin', choices=['lin', 'kl', 'w'],
                        help='distance function for discrepancy loss')
    parser.add_argument('--train_max_epoch', type=int, default=2000)  # 1000
    parser.add_argument('--epochs_ft', type=int, default=1000)
    parser.add_argument('--train_valid_interval', type=int, default=5)  # 5
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_lr', type=float, default=0.01)  # 0.01
    parser.add_argument('--lr_ft', type=float, default=5e-3)
    parser.add_argument('--train_min_lr', type=float, default=0.00001)  # 0.0001
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)  # 0.5
    parser.add_argument('--l2reg', type=float, default=5e-6)
    parser.add_argument('--lr_scheduler', type=str, default='zigzag', choices=['sgdr', 'cos', 'zigzag', 'none'],
                        help='lr scheduler')
    parser.add_argument('--train_decay_patience', type=int, default=50)  # 25
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)  # 50
    parser.add_argument('--top_n', type=int, default=6)
    args = parser.parse_args()

    devices = list(map(int, args.gpu.split(',')))

    run(args, devices)