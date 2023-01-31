import torch as th
from torch import nn
import dgl.function as fn
from tools.utils import get_activation
import torch.nn.functional as F
from geomloss import SamplesLoss
import math

def calc_disc(disc_func, user_embed, item_embed, nodepairs_f, nodepairs_cf):
    X_f = th.cat((user_embed[nodepairs_f.T[0]], item_embed[nodepairs_f.T[1]]), axis=1)
    X_cf = th.cat((user_embed[nodepairs_cf.T[0]], item_embed[nodepairs_cf.T[1]]), axis=1)
    if disc_func == 'lin':
        mean_f = X_f.mean(0)
        mean_cf = X_cf.mean(0)
        loss_disc = th.sqrt(F.mse_loss(mean_f, mean_cf) + 1e-6)
    elif disc_func == 'kl':
        # TODO: kl divergence
        pass
    elif disc_func == 'w':
        # Wasserstein distance
        dist = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        loss_disc = dist(X_cf, X_f)
    else:
        raise Exception('unsupported distance function for discrepancy loss')
    return loss_disc


class Encoder(nn.Module):
    def __init__(self, user_dim, item_dim, hide_dim, side_dim, act, layerNum,model_name):
        super(Encoder, self).__init__()
        self.layerNum = layerNum
        self.model_name = model_name
        self.ufc = nn.Linear(user_dim, hide_dim)
        self.ifc = nn.Linear(item_dim, hide_dim)
        self.side_fc = nn.Linear(side_dim, hide_dim)
        self.act = get_activation(act)

        self.layers = nn.ModuleList()
        for i in range(self.layerNum):
            self.layers.append(GCNLayer(feat_drop=0.7))

    def forward(self, graph, origin_user_embedding, origin_item_embedding, item_side_feat):
        userNum=len(origin_user_embedding)
        origin_user_embedding = self.ufc(origin_user_embedding)
        user_embedding = self.act(origin_user_embedding)
        mlp_user_embedding = user_embedding.clone()
        item_embedding=self.ifc(origin_item_embedding)
        item_embedding = self.act(item_embedding)
        if self.model_name=='mlp':
            return mlp_user_embedding, item_embedding

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, user_embedding, item_embedding)
            else:
                embeddings = layer(graph, embeddings[: userNum], embeddings[userNum: ])
            item_embedding = item_embedding + embeddings[userNum: ] * (1/(i+2))
            user_embedding  = user_embedding  + embeddings[:userNum] * (1 / (i + 2))

        if item_side_feat!=None:
            item_embedding = item_embedding + item_side_feat

        if self.model_name == "lightgcn":
            return user_embedding, item_embedding
        if self.model_name == "ddi":
            return mlp_user_embedding, item_embedding


class GCNLayer(nn.Module):
    def __init__(self,feat_drop):
        super(GCNLayer, self).__init__()
        self.feat_drop = nn.Dropout(feat_drop)

    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            node_f = th.cat([u_f, v_f], dim=0)
            node_f = self.feat_drop(node_f)
            # D^-1/2
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = th.pow(degs, -0.5).view(-1, 1)
            node_f = node_f * norm
            graph.ndata['n_f'] = node_f
            graph.update_all(message_func=fn.copy_src(src='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))
            rst = graph.ndata['n_f']
            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = th.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm
            return rst

class Decoder(nn.Module):
    def __init__(self, dec, dim_z, dim_h=64):
        super(Decoder, self).__init__()
        self.dec = dec
        if dec == 'innerproduct':
            dim_in = 2
        elif dec == 'hadamard':
            dim_in = dim_z + 1
        elif dec == 'mlp':
            dim_in = 1 + 2*dim_z
        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, dim_h, bias=True),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(dim_h, 1, bias=False)
        )

    def forward(self, z_i, z_j, T):
        if self.dec == 'innerproduct':
            z = (z_i * z_j).sum(1).view(-1, 1)
            h = th.cat((z, T.view(-1, 1)), dim=1)
        elif self.dec == 'mlp':
            h = th.cat((z_i, z_j, T.view(-1, 1)), dim=1)
        elif self.dec == 'hadamard':
            z = z_i * z_j
            h = th.cat((z, T.view(-1, 1)), dim=1)
        h = h.to(th.float32)
        h = self.mlp_out(h).squeeze()
        return h

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue

class MSGCN(nn.Module):
    def __init__(self,user_dim, item_dim, hid_dim, side_dim, act, layer_num, model_name, dec):
        super(MSGCN, self).__init__()
        self.encoder = Encoder(user_dim=user_dim, item_dim=item_dim, hide_dim=hid_dim, side_dim=side_dim,
                             act=act, layerNum=layer_num, model_name=model_name)
        self.decoder = Decoder(dec=dec,dim_z=hid_dim)

    def forward(self,train_g, user_features, item_features, item_side_feat, userId, pos_itemId, neg_itemId, T_f, T_cf):
        user_embed, item_embed = self.encoder(train_g, user_features, item_features, item_side_feat)
        userEmbed = user_embed[userId]
        posEmbed = item_embed[pos_itemId]
        negEmbed = item_embed[neg_itemId]
        userEmbed = th.cat((userEmbed, userEmbed),dim=0)
        itemEmbed = th.cat((posEmbed, negEmbed),dim=0)
        logits_f = self.decoder(userEmbed, itemEmbed, T_f)
        logits_cf = self.decoder(userEmbed, itemEmbed, T_cf)
        return user_embed, item_embed, logits_f, logits_cf


class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, lr_scheduler, *op):
        self.optimizers = op
        self.steps = 0
        self.reset_count = 0
        self.next_start_step = 10
        self.multi_factor = 2
        self.total_epoch = 0
        if lr_scheduler == 'sgdr':
            self.update_lr = self.update_lr_SGDR
        elif lr_scheduler == 'cos':
            self.update_lr = self.update_lr_cosine
        elif lr_scheduler == 'zigzag':
            self.update_lr = self.update_lr_zigzag
        elif lr_scheduler == 'none':
            self.update_lr = self.no_update

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
    def no_update(self, base_lr):
        return base_lr

    def update_lr_SGDR(self, base_lr):
        end_lr = 1e-3 # 0.001
        total_T = self.total_epoch + 1
        if total_T >= self.next_start_step:
            self.steps = 0
            self.next_start_step *= self.multi_factor
        cur_T = self.steps + 1
        lr = end_lr + 1/2 * (base_lr - end_lr) * (1.0 + math.cos(math.pi*cur_T/total_T))
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        self.total_epoch += 1
        return lr

    def update_lr_zigzag(self, base_lr):
        warmup_steps = 50
        annealing_steps = 20
        end_lr = 1e-4
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps+1) / warmup_steps
        elif self.steps < warmup_steps+annealing_steps:
            step = self.steps - warmup_steps
            q = (annealing_steps - step) / annealing_steps
            lr = base_lr * q + end_lr * (1 - q)
        else:
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr

    def update_lr_cosine(self, base_lr):
        """ update the learning rate of all params according to warmup and cosine annealing """
        # 400, 1e-3
        warmup_steps = 10
        annealing_steps = 500
        end_lr = 1e-3
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps+1) / warmup_steps
        elif self.steps < warmup_steps+annealing_steps:
            step = self.steps - warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / annealing_steps))
            lr = base_lr * q + end_lr * (1 - q)
        else:
            # lr = base_lr * 0.001
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr