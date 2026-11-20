import torch
from torch import nn
import torch.nn.functional as F
from .models import register_dependency_block
from .blocks import (TransformerBlock, MaskedConv1D, LayerNorm)
import numpy as np
import math

@register_dependency_block("DependencyBlock")
class Dependency_Block(nn.Module):
    """
        model co-occur and temporal dependency between events in a video.
    """
    def __init__(
        self,
        in_channel,      
        n_embd,
        n_embd_ks,
        num_classes,
        path_pdrop,     
        n_head = 1
    ):
        super().__init__()
        self.max_len = 256
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)

        self.feature_expand = MaskedConv1D(in_channel, self.num_classes,
                        n_embd_ks, stride=1, padding=n_embd_ks//2, bias=False)

        self.cooccur_branch = TransformerBlock(n_embd, n_head, n_hidden=n_embd, path_pdrop=path_pdrop)
        self.temporal_branch = nn.ModuleList()
        for idx in range(3):
            self.temporal_branch.append(TransformerBlock(self.num_classes, n_head, n_hidden=self.num_classes, path_pdrop=path_pdrop))

        self.feature_squeeze = MaskedConv1D(self.num_classes, in_channel, n_embd_ks,
                            stride=1, padding=n_embd_ks//2, bias=False)

         # init weights
        self.apply(self.__init_weights__)

        self.p_dropout = 0.5

        self.num_experts = 2
        self.gate = nn.ModuleList()
        self.gate.append(nn.Sequential(nn.Linear(self.num_classes, 64), nn.PReLU(), nn.Linear(64, self.num_experts)))

        self.experts = nn.ModuleList()
        self.experts.append(GCN_Block(self.max_len, self.p_dropout, bias=True, node_n=100, non_dia=0.01, dia=0))
        self.experts.append(GCN_Block(self.max_len, self.p_dropout, bias=True, node_n=100, non_dia=0, dia=1))
        self.experts.append(GCN_Block(self.max_len, self.p_dropout, bias=True, node_n=100, non_dia=0.01, dia=0.99))
        self.experts.append(GCN_Block(self.max_len, self.p_dropout, bias=True, node_n=100, non_dia=0.01, dia=0.99))
        self.experts.append(GCN_Block(self.max_len, self.p_dropout, bias=True, node_n=100, non_dia=0.01, dia=0))
        self.experts.append(GCN_Block(self.max_len, self.p_dropout, bias=True, node_n=100, non_dia=0, dia=1))
        self.experts.append(GCN_Block(self.max_len, self.p_dropout, bias=True, node_n=100, non_dia=0.01, dia=0.99))
        self.experts.append(GCN_Block(self.max_len, self.p_dropout, bias=True, node_n=100, non_dia=0.01, dia=0.99))

        self.proj = nn.Sequential(nn.Linear(self.num_classes, self.num_classes), nn.PReLU(), nn.Linear(self.num_classes, self.num_classes))

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def gumbel_softmax(self, logits, tau=0.1, hard=False, eps=1e-10):
        # Get Gumbel noise
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            # Straight through.
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, fpn_feats, fpn_masks, middle_feats_V, middle_feats_A):
        assert len(fpn_feats) == len(fpn_masks)
        moe_feats = tuple()
        out_features_list = tuple()
        gate_cls = tuple()
        for idx, (features, mask, feats_V, feats_A) in enumerate(zip(fpn_feats, fpn_masks, middle_feats_V, middle_feats_A)):

            features_exp, mask = self.feature_expand(features, mask)
            features_exp = features_exp.permute(0, 2, 1)
            features_exp = self.relu(features_exp)
            B, T, C = features_exp.size()     #b*Ti*100

            features_exp = features_exp.permute(0, 2, 1).contiguous()
            features_exp_t, _ = self.temporal_branch[0](features_exp, features_exp, mask)     # b*100*Ti
            features_exp_t, _ = self.temporal_branch[1](features_exp_t, features_exp_t, mask)
            features_exp = features_exp_t.permute(0, 2, 1).contiguous()

            features_exp = features_exp.view(B, T, C).contiguous()
            for ei in range(4):
                expert_outputs = torch.stack([expert(features_exp) for expert in self.experts[ei*self.num_experts:(ei+1)*self.num_experts]], dim=1)
                expert_outputs = expert_outputs.permute(0, 3, 1, 2).contiguous().view(-1, self.num_experts, C).contiguous()    # BTi*n*C
                features_exp = features_exp.view(-1, C).contiguous()
                gate_score = self.gate[0](features_exp)
                # gate_score = self.gate[ei](features_exp)
                gate_score = self.gumbel_softmax(gate_score, 0.05, True)
                # gate_score = F.softmax(gate_score, dim=-1)
                features_exp = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)       # BTi*C
                features_exp = features_exp.view(B, T, C).contiguous()     # B*T*C

            moe_features = features_exp.view(-1, C).contiguous()
            moe_features = self.proj(moe_features).view(B, T, C).contiguous()
            moe_features = moe_features.permute(0, 2, 1).contiguous()

            features_exp_t, _ = self.temporal_branch[2](features_exp_t, features_exp_t, mask)
            out_features = features_exp_t + moe_features
            out_features, mask = self.feature_squeeze(out_features, mask)    # B*1024*T

            gate_score = gate_score.view(B, -1, self.num_experts).contiguous()

            moe_feats += (moe_features,)
            out_features_list += (out_features,)
            gate_cls += (gate_score, )

        return out_features_list, fpn_masks, gate_cls, moe_feats


class GCN_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=100, non_dia=0.01, dia=0.99):
        """
        Define a residual block of GCN
        """
        super(GCN_Block, self).__init__()
        self.non_dia = non_dia
        self.dia = dia
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, non_dia=self.non_dia, dia=self.dia)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):

        y = self.gc2(x)
        y = y.contiguous()
        b, c, t = y.shape
        y = y.view(b, c, t).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        return y + x.permute(0, 2, 1)


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=100, non_dia=0.01, dia=0.99):
        super(GraphConvolution, self).__init__()
        self.in_features = None
        self.out_features = None
        # self.weight = nn.Linear(in_features, out_features)
        self.att_weight = nn.Parameter(torch.FloatTensor(non_dia + dia * np.eye(node_n)[np.newaxis, ...]))

        self.conv = nn.Conv1d(in_channels=node_n, out_channels=node_n, kernel_size=3, padding=1)
        self.linear = nn.Linear(out_features, out_features)
        self.act = nn.LeakyReLU(0.1)
        self.do = nn.Dropout(0.1)

    def forward(self, x):

        in_features = x.size(1)
        out_features = x.size(1)
        y = self.conv(x.permute(0, 2, 1))      #B*C*T
        gatt = torch.matmul(self.att_weight, y)       #B*C*T
        output = gatt

        return output