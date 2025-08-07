import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class GatedFusionV1(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim*4, in_dim),
            nn.PReLU(),
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, h1, h2):
        diff = torch.abs(h1 - h2)
        prod = h1 * h2
        gate = self.gate(torch.cat([h1, h2, diff, prod], dim=-1))
        return gate * h1 + (1 - gate) * h2

class GatedFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, h1, h2):
        t_1 = self.gate1(h1)
        t_2 = self.gate2(h2)
        out  = torch.cat([t_1 * h1, t_2 * h2], dim=-1)
        return out

class GCN(nn.Module):

    def __init__(self, in_feats, h_feats, dropout):
        super(GCN, self).__init__()
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        #PReLU和ReLU、GELU、ELU        BatchNorm1d和LayerNorm
        self.conv_1 = GraphConv(in_feats, h_feats, activation=nn.ELU(), norm='both')
        self.bn1 = nn.BatchNorm1d(h_feats)

        self.conv_2 = GraphConv(h_feats, h_feats, activation=nn.ELU(), norm='both')
        self.bn2 = nn.BatchNorm1d(h_feats)

    def forward(self, graph, features):
        h1 = self.dropout_1(features)
        h1 = self.conv_1(graph, h1)
        h1 = self.bn1(h1)
        h2 = self.dropout_2(h1)
        h2 = self.conv_2(graph, h2)
        h2 = self.bn2(h2)
        return h2

class MixModel(nn.Module):
    """混合模型，仅使用两个社交网络图（mention_graph 和 retweet_graph）"""



    def __init__(self, num_classes=129, hidden_dim=0, in_feats=0, dropout=0.):
        super(MixModel, self).__init__()
        # 社交关系 GCN
        self.social_gcn = GCN(in_feats=in_feats, h_feats=hidden_dim, dropout=dropout)

        # 转发关系 GCN
        self.retweet_gcn = GCN(in_feats=in_feats, h_feats=hidden_dim, dropout=dropout)

        # # 门控融合机制、注意力机制融合
        # self.fusion = GatedFusion(hidden_dim)
        self.fusion = GatedFusionV1(hidden_dim)
        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=num_classes)
        )

    def forward(self, mention_graph, retweet_graph, node_features):

        # 处理社交关系图
        h2 = self.social_gcn(mention_graph, node_features)

        # 处理转发关系图
        h3 = self.retweet_gcn(retweet_graph, node_features)

        # 使用门控融合机制融合两个图的特征
        combined_features = self.fusion(h2, h3)

        # 拼接图的特征
        # combined_features = torch.cat([h2, h3], dim=-1)

        # 通过全连接层转换为最终分类 logits
        output = self.fc(combined_features)

        return output
