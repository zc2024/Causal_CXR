import os, sys
import os.path as osp
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
from collections import OrderedDict
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.DNet import DModule

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class GroupWiseLinear1(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class GroupWiseLinear2(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Attention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.relu(x1 + x2 + x3)
        return y


class Causal(nn.Module):
    def __init__(self, backbone, transfomer, num_class,bs):

        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        ekernel_size = 3
        self.sigmoid = nn.Sigmoid()
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ####
            nn.Conv1d(1, 1, kernel_size=ekernel_size, padding=(ekernel_size - 1) // 2),
            nn.Sigmoid(),
            ###
        )

        self.bs = bs
        dowmsphw = 26
        self.hw = dowmsphw
        hidden_dim = transfomer.d_model
        self.dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)
        self.fc_add = GroupWiseLinear1(num_class, hidden_dim, bias=True)
        self.fc_cat = GroupWiseLinear2(num_class, hidden_dim*2, bias=True)
        self.dnet=DModule(d_model=hidden_dim,kernel_size=3,H=dowmsphw,W=dowmsphw)
        self.conv = nn.Conv1d(2 * hidden_dim, hidden_dim,1)

        self.att = Attention(channel=2048)
        self.cat_or_add = "cat"

    def forward(self, input):

        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        src0 = self.dnet(src)
        src00 = self.seq(src)
        src1 = src00.flatten(2) + self.att(src).flatten(2)
        src2 = torch.cat((src0.flatten(2), src1), 1)
        src = self.conv(src2).reshape(self.bs, self.dim, self.hw, self.hw)
        query_input = self.query_embed.weight

        #To get two outputs of Transformer, we use modified nn.functional.multi_head_attention_forward
        # ToDo: Migrate source code changes(nn.functional.multi_head_attention_forward) to transformer.py
        hs = self.transformer(self.input_proj(src), query_input, pos)[0]  # B,K,d ([1, bs, 15, 2048])
        useless = self.transformer(self.input_proj(src), query_input, pos)[1]  # B,K,d
        hs = hs[-1]
        useless = useless[-1]

        num = hs.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.cat_or_add == "cat":
            halfuse = torch.cat((useless[random_idx], hs), dim=2)
            halfout = self.fc_cat(halfuse)
        else:
            halfuse = useless[random_idx] + hs
            halfout = self.fc_add(halfuse)

        out = self.fc(hs)

        # import ipdb; ipdb.set_trace()
        return out, halfout

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))


def build_net(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Causal(
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_class,
        bs = args.batch_size
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")

    return model
