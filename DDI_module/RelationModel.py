import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import pandas as pd
from dgl.nn import SAGEConv, GINConv
import dgl.function as fn
from DDI_module.genKGembd import getEmbd



class Reg_loss(nn.Module):
    def __init__(self,dev):
        super().__init__()
        self.dev=dev
        self.compute_cosloss = CosLoss()
        self.pred = DotPredictor()

    def forward(self, id_list, item_feat):
        DBID2NodeIndex_dic = {ele: i for i, ele in enumerate(id_list)}
        compd_relations = pd.read_excel(
            '/data/compound_relations_syn&ant_1v1.xlsx')
        compd_relations = compd_relations[compd_relations.iloc[:, 0].isin(id_list)]
        compd_relations = compd_relations[compd_relations.iloc[:, 1].isin(id_list)]
        edge_row = compd_relations.iloc[:, 0].values.tolist()
        edge_col = compd_relations.iloc[:, 1].values.tolist()
        edge_type = compd_relations.iloc[:, 2]

        src = np.array([DBID2NodeIndex_dic[edgeID] for edgeID in edge_row])
        dst = np.array([DBID2NodeIndex_dic[edgeID] for edgeID in edge_col])
        labels = list(edge_type)
        g = dgl.graph((src, dst)).to(self.dev)
        g.ndata['h'] = item_feat
        # 算cos loss
        cos_loss = self.compute_cosloss(g, labels)
        # 算mse loss
        pos_u, pos_v = g.edges()
        pos1_index = [i for i, x in enumerate(labels) if x is 1]
        neg1_index = [i for i, x in enumerate(labels) if x is -1]
        pos1_u, pos1_v = pos_u[pos1_index], pos_v[pos1_index]
        neg1_u, neg1_v = pos_u[neg1_index], pos_v[neg1_index]
        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(pos_u)), (pos_u.cpu().numpy(), pos_v.cpu().numpy())))
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)
        neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
        neg_u, neg_v = neg_u[neg_eids], neg_v[neg_eids]
        pos1_g = dgl.graph((pos1_u, pos1_v), num_nodes=g.number_of_nodes())
        neg1_g = dgl.graph((neg1_u, neg1_v), num_nodes=g.number_of_nodes())
        neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())
        pos1_score = self.pred(pos1_g.to(self.dev), item_feat)
        neg1_score = self.pred(neg1_g.to(self.dev), item_feat)
        neg_score = self.pred(neg_g.to(self.dev), item_feat)
        mseloss = compute_mseloss(pos1_score, neg1_score, neg_score)
        print('cos, mse:', cos_loss.item(), mseloss.item())
        # loss=cos_loss+mseloss
        loss = cos_loss
        return loss

class CosLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, g, label):
        with g.local_scope():
            # g.ndata['h'] = h
            src,dst = g.edges()
            src_feat = g.ndata['h'][src]
            dst_feat = g.ndata['h'][dst]
            output = F.cosine_similarity(src_feat, dst_feat, dim=1)
            pos_pred=[]
            neg_pred=[]
            for i,y in enumerate(label):
                if y ==1:
                    pos_pred.append(output[i])
                else:
                    neg_pred.append(output[i])
            pos_loss = -th.log(th.tensor(pos_pred).clamp_min(1e-15)).mean()
            neg_loss = -th.log((-th.tensor(neg_pred)).clamp_min(1e-15)).mean()
            cos_loss=pos_loss+neg_loss
            return cos_loss.requires_grad_(True)


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = th.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h

class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = th.nn.ModuleList()
            self.batch_norms = th.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, learn_eps,neighbor_pooling_type):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = th.nn.ModuleList()
        self.batch_norms = th.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
        return h

def compute_mseloss(pos1_score, neg1_score, neg_score):
    scores = th.cat([pos1_score, neg1_score, neg_score])
    labels = th.cat([th.ones(pos1_score.shape[0]), th.ones(neg1_score.shape[0])*-1,
                     th.zeros(neg_score.shape[0])]).to(scores.device)
    return F.mse_loss(scores, labels)


def train_all(g, labels, item_side_units):
    pos_u, pos_v = g.edges()
    pos1_index = [i for i, x in enumerate(labels) if x is 1]
    neg1_index = [i for i, x in enumerate(labels) if x is -1]

    pos1_u, pos1_v = pos_u[pos1_index], pos_v[pos1_index]
    neg1_u, neg1_v = pos_u[neg1_index], pos_v[neg1_index]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(pos_u)), (pos_u.numpy(), pos_v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    neg_u, neg_v = neg_u[neg_eids], neg_v[neg_eids]

    pos1_g = dgl.graph((pos1_u, pos1_v), num_nodes=g.number_of_nodes())
    neg1_g = dgl.graph((neg1_u, neg1_v), num_nodes=g.number_of_nodes())
    neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())


    # model = GraphSAGE(g.ndata['feat'].shape[1],item_side_units)
    model = GIN(num_layers=3,num_mlp_layers=2,input_dim=g.ndata['feat'].shape[1],
                hidden_dim=item_side_units,learn_eps=True,neighbor_pooling_type='mean')
    # You can replace DotPredictor with MLPPredictor.
    # pred = MLPPredictor(item_side_units)
    compute_cosloss=CosLoss()
    pred = DotPredictor()
    optimizer = th.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.001) #0.001
    for e in range(400):#350
        h = model(g, g.ndata['feat'])
        pos1_score = pred(pos1_g, h)
        neg1_score = pred(neg1_g, h)
        neg_score = pred(neg_g, h)
        mseloss = compute_mseloss(pos1_score, neg1_score, neg_score)
        g.ndata['h'] = h
        cosloss = compute_cosloss(g, labels)
        total_loss=mseloss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # if e % 1 == 0:
            # print('In epoch {}, mseloss: {}, cosloss: {}'.format(e, mseloss,cosloss))
    return h.data


def genRelationEmbd(id_list,item_side_units,type):
    DBID2NodeIndex_dic = {ele:i for i,ele in enumerate(id_list)}
    if type=='pretrain':
        origin_embd = getEmbd(id_list)
        node_features = th.tensor(origin_embd)
    if type=='onehot':
        drug_one_hot = np.zeros(shape=(len(id_list), len(id_list)), dtype=np.float32)
        drug_one_hot[np.arange(len(id_list)),np.array([DBID2NodeIndex_dic[ele] for ele in id_list])] = 1
        node_features = th.tensor(drug_one_hot)

    compd_relations = pd.read_excel('/data/compound_relations_syn&ant_1v1.xlsx')
    compd_relations = compd_relations[compd_relations.iloc[:, 0].isin(id_list)]
    compd_relations = compd_relations[compd_relations.iloc[:, 1].isin(id_list)]
    edge_row = compd_relations.iloc[:,0].values.tolist()
    edge_col = compd_relations.iloc[:,1].values.tolist()
    temp_row,temp_col=[],[]
    temp_row.extend(edge_row)
    temp_row.extend(edge_col)
    temp_col.extend(edge_col)
    temp_col.extend(edge_row)
    edge_type = compd_relations.iloc[:,2]
    # print(len(edge_row),len(edge_col),len(edge_type[edge_type.iloc[:]==1]),len(edge_type[edge_type.iloc[:]==-1]))

    src = np.array([DBID2NodeIndex_dic[edgeID] for edgeID in temp_row])
    dst = np.array([DBID2NodeIndex_dic[edgeID] for edgeID in temp_col])
    labels = list(edge_type)
    labels.extend(labels)

    relation=[]
    for i in range(len(src)):
        # relation.append([src[i],dst[i],labels[i]])
        relation.append([src[i], dst[i]])
    df = pd.DataFrame(data=relation,
                      columns= ['drug1','drug2'])
    df = df.loc[~(df['drug1']==df['drug2'])]
    # df = df.drop_duplicates(['drug1','drug2']) #-1: 486->481 1:194->193 有问题
    df = df + 1
    df.sort_values(by=['drug1', 'drug2'], ascending=[True, True], ignore_index=True, inplace=True)
    g = dgl.graph((src,dst))
    g.ndata['feat'] = node_features

    emb_list=train_all(g,labels,item_side_units)
    return emb_list