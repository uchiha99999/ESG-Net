import torch
from torch import nn
from torch.nn import functional as F

from .models import register_multimodal_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, 
                    MaskedConv1D, LayerNorm)


@register_multimodal_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in_V,                # input visual feature dimension
        n_in_A,                # input audio feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        scale_factor = 2,      # dowsampling rate for the branch,
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        # print(n_in_V, n_in_A, n_embd, n_head)   # 2048 128 512 4
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd_V = nn.ModuleList()
        self.embd_A = nn.ModuleList()
        self.embd_norm_V = nn.ModuleList()
        self.embd_norm_A = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels_V = n_in_V
                in_channels_A = n_in_A
            else:
                in_channels_V = n_embd
                in_channels_A = n_embd
            self.embd_V.append(MaskedConv1D(
                    in_channels_V, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            self.embd_A.append(MaskedConv1D(
                    in_channels_A, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm_V.append(
                    LayerNorm(n_embd)
                )
                self.embd_norm_A.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm_V.append(nn.Identity())
                self.embd_norm_A.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.self_att_V = nn.ModuleList()
        self.self_att_A = nn.ModuleList()
        self.self_att_ALL_dim = nn.ModuleList()
        self.self_att_ALL = nn.ModuleList()

        for idx in range(arch[1]-1):
            self.self_att_V.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.self_att_A.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        for idz in range(arch[1] - 1):
            self.self_att_ALL.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                )
            )


        self.pro1 = nn.Sequential(nn.Linear(512, 512))
        self.pro2 = nn.Sequential(nn.Linear(512, 512))
        self.pro3 = nn.Sequential(nn.Linear(512, 640), nn.PReLU(), nn.Linear(640, 1024))
        self.pro4 = nn.Sequential(nn.Linear(512, 640), nn.PReLU(), nn.Linear(640, 1024))
        self.proj3 = nn.Sequential(nn.Linear(512, 640), nn.PReLU(), nn.Linear(640, 1024))
        self.proj4 = nn.Sequential(nn.Linear(512, 640), nn.PReLU(), nn.Linear(640, 1024))
        self.project3 = nn.Sequential(nn.Linear(512, 640), nn.PReLU(), nn.Linear(640, 1024))
        self.project4 = nn.Sequential(nn.Linear(512, 640), nn.PReLU(), nn.Linear(640, 1024))
        self.sencoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.max_len, nhead=8, dropout=0.1), num_layers=2)

        #cross-attention on original temporal resolution
        self.ori_cross_att_Va = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
        self.ori_cross_att_Av = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )

        self.ori_cross_att_V_all = TransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
        )
        self.ori_cross_att_A_all = TransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
        )

        #cross-attention after down-sampling
        self.cross_att_Va = nn.ModuleList()
        self.cross_att_Av = nn.ModuleList()
        for idx in range(arch[2]):
            self.cross_att_Va.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.cross_att_Av.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        # init weights
        self.apply(self.__init_weights__)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_V, x_A, mask):
        # x_V/x_A: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C_V, T = x_V.size()
        mask_V = mask_A = mask
        #print("emb1", x_V.shape, x_A.shape)

        # embedding network
        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V)
            x_V = self.relu(self.embd_norm_V[idx](x_V))
            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd

            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)


        # stem transformer
        x_1 = x_V * mask_V
        x_2 = x_A * mask_A
        x_1 = x_1.permute(0, 2, 1).reshape(-1, x_V.size(1))
        x_2 = x_2.permute(0, 2, 1).reshape(-1, x_A.size(1))
        x_1 = self.pro1(x_1).reshape(B, x_V.size(2), x_V.size(1))
        x_2 = self.pro2(x_2).reshape(B, x_A.size(2), x_A.size(1))
        x_V = x_1.permute(0, 2, 1)
        x_A = x_2.permute(0, 2, 1)
        x_ALL_res = x_1 * x_2
        x_res = x_ALL_res.permute(2, 0, 1)
        x_ALL = self.sencoder(x_res).permute(1, 2, 0)
        x_ALL = x_ALL + x_ALL_res
        x_ALL = x_ALL.permute(0, 2, 1)

        mask_ALL = mask
        #mask_ALL_dim = torch.ones((B, 1, 1024), dtype=torch.bool, device=mask.device)

        for idx in range(len(self.self_att_V)):
            x_V_res = x_V
            x_A_res = x_A
            x_V, mask_V = self.self_att_V[idx](x_V, x_V, mask_V)
            x_A, mask_A = self.self_att_A[idx](x_A, x_A, mask_A)
            x_V += x_V_res
            x_A += x_A_res

        x_Vcc = x_V.permute(0, 2, 1).reshape(-1, x_V.size(1))
        x_Acc = x_A.permute(0, 2, 1).reshape(-1, x_A.size(1))
        x_Vcc = self.project3(x_Vcc).reshape(B, x_V.size(2), -1).permute(0, 2, 1)
        x_Acc = self.project4(x_Acc).reshape(B, x_A.size(2), -1).permute(0, 2, 1)
        middle_V = tuple()
        middle_A = tuple()
        middle_V += (x_Vcc, )
        middle_A += (x_Acc, )
        for i in range(5):
            x_Vcc = self.avg_pool(x_Vcc)
            middle_V += (x_Vcc, )
            x_Acc = self.avg_pool(x_Acc)
            middle_A += (x_Acc, )

        for idx in range(len(self.self_att_V)):
            x_ALL_res = x_ALL
            x_ALL, mask_ALL = self.self_att_ALL[idx](x_ALL, x_ALL, mask_ALL)
            x_ALL += x_ALL_res

        x_ALL1 = x_ALL
        x_ALL2 = x_ALL
        x_Va, mask_V = self.ori_cross_att_V_all(x_V, x_ALL1, mask_V)
        x_Av, mask_V = self.ori_cross_att_A_all(x_A, x_ALL2, mask_A)


        Vcc = x_Va.permute(0, 2, 1).reshape(-1, x_V.size(1))
        Acc = x_Av.permute(0, 2, 1).reshape(-1, x_A.size(1))
        Vcc = self.proj3(Vcc).reshape(B, x_V.size(2), -1).permute(0, 2, 1)
        Acc = self.proj4(Acc).reshape(B, x_A.size(2), -1).permute(0, 2, 1)
        mid_V = tuple()
        mid_A = tuple()
        mid_V += (Vcc, )
        mid_A += (Acc, )
        for i in range(5):
            Vcc = self.avg_pool(Vcc)
            mid_V += (Vcc, )
            Acc = self.avg_pool(Acc)
            mid_A += (Acc, )

        x_Va, mask_V = self.ori_cross_att_Va(x_Va, x_A, mask_V)
        x_Av, mask_V = self.ori_cross_att_Av(x_Av, x_V, mask_A)

        # prep for outputs
        out_feats_V = tuple()
        out_feats_A = tuple()
        out_masks_V = tuple()
        out_masks_A = tuple()
        # 1x resolution
        out_feats_V += (x_Va, )
        out_masks_V += (mask_V, )
        out_feats_A += (x_Av, )
        out_masks_A += (mask_A, )

        x_Vc = x_Va.permute(0, 2, 1).reshape(-1, x_Va.size(1))
        x_Ac = x_Av.permute(0, 2, 1).reshape(-1, x_Av.size(1))
        x_Vc = self.pro3(x_Vc).reshape(B, self.max_len, -1).permute(0, 2, 1)
        x_Ac = self.pro4(x_Ac).reshape(B, self.max_len, -1).permute(0, 2, 1)

        middle_feats_V = tuple()
        middle_feats_A = tuple()

        middle_feats_V += (x_Vc, )
        middle_feats_A += (x_Ac, )

        # main branch with downsampling
        for idx in range(len(self.cross_att_Va)):
            x_V, mask_V = self.cross_att_Va[idx](out_feats_V[idx], out_feats_A[idx], mask_V)
            out_feats_V += (x_V, )
            out_masks_V += (mask_V, )
            x_Vc = x_V.permute(0, 2, 1).reshape(-1, x_V.size(1))
            x_Vc = self.pro3(x_Vc).reshape(B, x_V.size(2), -1).permute(0, 2, 1)
            middle_feats_V += (x_Vc, )

            x_A, mask_A = self.cross_att_Av[idx](out_feats_A[idx], out_feats_V[idx], mask_A)
            out_feats_A += (x_A, )
            out_masks_A += (mask_A, )
            x_Ac = x_A.permute(0, 2, 1).reshape(-1, x_A.size(1))
            x_Ac = self.pro4(x_Ac).reshape(B, x_A.size(2), -1).permute(0, 2, 1)
            middle_feats_A += (x_Ac, )

        return out_feats_V, out_feats_A, out_masks_V, middle_V, middle_A, mid_V, mid_A, middle_feats_V, middle_feats_A