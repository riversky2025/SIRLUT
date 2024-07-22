import time

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
import trilinear
import kornia
import itertools
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops.layers.torch import Rearrange
from einops import rearrange

def base_block(in_filters, out_filters, normalization=False, kernel_size=3, stride=2, padding=1):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))

    return layers
class ChannelAttention(nn.Module):
    def __init__(self, embed_dim,device, num_chans=4, expan_att_chans=2, fusion=False):
        super(ChannelAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.isFution = fusion

        self.group_qkv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim),
            Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * 3),
        )
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x, text):

        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).contiguous().chunk(3, dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t

        x_ = attn.softmax(dim=-1) @ v
        x_ = rearrange(x_, "B C X (H W)-> B (X C) H W", B=B, W=W, H=H, C=self.num_heads).contiguous()

        x_ = self.group_fus(x_)
        return x_
class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(dim, int(dim / 2), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, int(dim / 2), kernel_size=3, stride=1, padding=1)
        # self.end = nn.Sequential(
        #     nn.Conv2d(dim * 4, dim * 2, kernel_size=3, stride=1, padding=1, groups=2),
        #     nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        # )

    def forward(self, rgb, other):
        x, y = self.conv1(rgb), self.conv2(other)
        x = torch.cat([x, y], dim=1)
        return F.relu(x)
class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_chans=2, expan_att_chans=1):
        super(SpatialAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        
        self.kConv = BasicBlock(self.num_heads * 2, self.num_heads)
        self.vConv = BasicBlock(self.num_heads * 2, self.num_heads)
        self.ch_radio = 1
        self.qkvexten = 2

        self.t = nn.Parameter(torch.ones(1, self.num_heads * self.ch_radio, 1, 1))
        self.t2 = nn.Parameter(torch.ones(1, self.num_heads * self.ch_radio, 1, 1))

        self.group_qkv_rgb = nn.Sequential(
            BasicBlock(embed_dim, embed_dim * expan_att_chans * 3, kernel_size=3, groups=embed_dim,norm=True),
            Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * 3),
        )
        # if not att_share:
        #     self.group_qkv_ir = nn.Sequential(
        #         BasicBlock(embed_dim, embed_dim * expan_att_chans * self.qkvexten, 3, groups=embed_dim,norm=True),
        #         Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * self.qkvexten),
        #     )
       

        self.group_fus = BasicBlock(embed_dim * expan_att_chans * self.ch_radio, embed_dim, 3, groups=embed_dim)

    def attnfun(self, q, k, v, t):
        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = q.transpose(-2, -1) @ k * t
        return (attn.softmax(dim=-1) @ v.transpose(-2, -1)).transpose(-2, -1).contiguous()

    def forward(self, x, ir):
        B, C, H, W = x.size()
        C_exp = self.expan_att_chans * C
        qv, kv, vv = self.group_qkv_rgb(x).contiguous().chunk(3, dim=1)
        qv = qv.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        kv = kv.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        vv = vv.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        
        qir, kir, vir = self.group_qkv_rgb(ir).contiguous().chunk(3, dim=1)
        qir = qir.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        
        kir = kir.view(B, self.num_heads , C_exp // self.num_heads, H * W)
        vir = vir.view(B, self.num_heads , C_exp // self.num_heads, H * W)
        
        kir=self.kConv(torch.cat([kv,kir],dim=1))
        vir=self.vConv(torch.cat([vv,vir],dim=1))
        x_ = self.attnfun(qv, kv, vv, self.t)
        x_ = self.attnfun(x_, kir, vir, self.t2)
        x_ = rearrange(x_, "B C X (H W)-> B (X C) H W", B=B, W=W, H=H).contiguous()
        x_ = self.group_fus(x_)
        return x_
class CnnFusionRs(nn.Module):
    def __init__(self, embed_dim,num_chans=8, expan_att_chans=1, fusion=True):
        super(CnnFusionRs, self).__init__()

        self.cnn = nn.Sequential(
            *base_block(embed_dim * 2, embed_dim, stride=1, normalization=True),
        )
        self.block1 = nn.Sequential(
            *base_block(embed_dim, embed_dim * 2, stride=1, normalization=True),
            *base_block(embed_dim * 2, embed_dim, stride=1, normalization=True),
        )
        self.block2 = nn.Sequential(
            *base_block(embed_dim, embed_dim * 2, stride=1, normalization=True),
            *base_block(embed_dim * 2, embed_dim, stride=1, normalization=True),
        )

    def init_weight(self):
        for l in self.cnn.modules():
            if isinstance(l, nn.Conv2d):
                xavier_init(l, gain=0.01)

    def forward(self, x, text):
        x_ = torch.cat([x, text], dim=1)
        x_ = self.cnn(x_)
        x = self.block1(x_) + x_
        x = self.block2(x) + x
        return x


class ChannelAttentionRs(nn.Module):
    def __init__(self, embed_dim, num_chans=2, expan_att_chans=1, lead_strady='cat', att_share=True, fusion=True):
        super(ChannelAttentionRs, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        if lead_strady == 'cat':
            self.ch_radio = 1
            self.qkvexten=3
            self.qConv = BasicBlock(self.num_heads * 2, self.num_heads)
            self.kConv = BasicBlock(self.num_heads * 2, self.num_heads)
            self.vConv = BasicBlock(self.num_heads * 2, self.num_heads)
        else:
            if lead_strady=='double-co':
                self.kConv = BasicBlock(self.num_heads * 2, self.num_heads)
                self.vConv = BasicBlock(self.num_heads * 2, self.num_heads)
            self.ch_radio = 1
            self.qkvexten = 2

        self.t = nn.Parameter(torch.ones(1, self.num_heads * self.ch_radio, 1, 1))
        self.t2 = nn.Parameter(torch.ones(1, self.num_heads * self.ch_radio, 1, 1))
        self.isFution = fusion
        self.att_share = att_share
        self.group_qkv_rgb = nn.Sequential(
            BasicBlock(embed_dim, embed_dim * expan_att_chans * 3, kernel_size=3, groups=embed_dim,norm=True),
            Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * 3),
        )
        if not att_share:
            self.group_qkv_ir = nn.Sequential(
                BasicBlock(embed_dim, embed_dim * expan_att_chans * self.qkvexten, 3, groups=embed_dim,norm=True),
                Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * self.qkvexten),
            )
        self.lead_strady = lead_strady

        self.group_fus = BasicBlock(embed_dim * expan_att_chans * self.ch_radio, embed_dim, 3, groups=embed_dim)

    def attnfun(self, q, k, v, t):
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * t
        return attn.softmax(dim=-1) @ v

    def forward(self, x, ir):
        B, C, H, W = x.size()
        C_exp = self.expan_att_chans * C
        qv, kv, vv = self.group_qkv_rgb(x).contiguous().chunk(3, dim=1)
        qv = qv.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        kv = kv.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        vv = vv.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        if self.att_share:
            qir, kir, vir = self.group_qkv_rgb(ir).contiguous().chunk(3, dim=1)
            qir = qir.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        else:
            if self.lead_strady=='co':
                kir, vir = self.group_qkv_ir(ir).contiguous().chunk(2, dim=1)
            else:
                qir, kir, vir = self.group_qkv_ir(ir).contiguous().chunk(3, dim=1)
                qir = qir.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        kir = kir.view(B, self.num_heads , C_exp // self.num_heads, H * W)
        vir = vir.view(B, self.num_heads , C_exp // self.num_heads, H * W)
        if self.lead_strady == 'cat':
            q, k, v = self.qConv(torch.cat([qv, qir], dim=1)),self.kConv( torch.cat([kv, kir], dim=1)), self.vConv(torch.cat([vv, vir], dim=1))
            x_ = self.attnfun(q, k, v, self.t)
        else:
            if self.lead_strady=="double-co":
                kir=self.kConv(torch.cat([kv,kir],dim=1))
                vir=self.vConv(torch.cat([vv,vir],dim=1))
            x_ = self.attnfun(qv, kv, vv, self.t)
            x_ = self.attnfun(x_, kir, vir, self.t2)
        x_ = rearrange(x_, "B C X (H W)-> B (X C) H W", B=B, W=W, H=H).contiguous()
        x_ = self.group_fus(x_)
        return x_
class IRBackbone(nn.Module):
    # (360,540) (480,720) 448 448
    def __init__(self, device,lead_strady,n_lut=3, input_resolution=(512,512) ,expan_att_chans=4,press_radio=8,ch_radio=8,att_share=True, share_conv=False):
        super().__init__()
        ya,ch=press_radio,ch_radio
        dim=16
        dim_2=32
        ch_input=int(dim * ch * ch / ya)
        self.share_conv=share_conv
        self.head=nn.Sequential(
            # BasicBlock(3, dim, norm=True),
            nn.Upsample(size=input_resolution, mode='bilinear'),
            nn.Conv2d(3, dim_2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(dim_2, affine=True),
            *discriminator_block(dim_2, dim,stride=2, normalization=True),
            # *discriminator_block(32, 64, stride=4,normalization=True),
            # *discriminator_block(64, 128, normalization=True),
            # *discriminator_block(128, 128),
            # nn.Upsample(size=input_resolution, mode='bilinear'),
            nn.PixelUnshuffle(ch_radio),
            BasicBlock(dim*ch_radio*ch_radio, ch_input, norm=True),

        )
        self.headText=nn.Sequential(
            nn.Upsample(size=input_resolution, mode='bilinear'),
            nn.Conv2d(3, dim_2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(dim_2, affine=True),
            *discriminator_block(dim_2, dim, stride=2, normalization=True),
            # *discriminator_block(32, 64,stride=4, normalization=True),
            # *discriminator_block(64, 128, normalization=True),
            # *discriminator_block(128, 128),
            # nn.Upsample(size=input_resolution, mode='bilinear'),
            nn.PixelUnshuffle(ch_radio),
            BasicBlock(dim*ch_radio*ch_radio, ch_input,  norm=True),
        )
        if lead_strady == 'sa':
            self.body = SpatialAttention(ch_input,expan_att_chans=expan_att_chans)
        elif lead_strady == 'cnn':
            self.body=CnnFusionRs(ch_input)
        else :
            self.body =ChannelAttentionRs(ch_input,expan_att_chans=expan_att_chans,lead_strady=lead_strady,att_share=att_share)
        self.body.eval()
        self.head.eval()
        self.headText.eval()



        self.tail2 = nn.Sequential(
            # BasicBlock(int(dim * ch * ch / ya), 128,stride=2),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(2),
            BasicBlock(ch_input, ch_input, stride=2),
        )
        self.liner = nn.Sequential(
            nn.Linear(ch_input, n_lut),
        )
        self.input_resolution = input_resolution
        self.out_channels = 128 * 4

    def forward(self, imgs,ir):
        _,_,W,H=imgs.shape
        # self.input_resolution=tuple(sorted(self.input_resolution))
        # if W>H:
        #     self.input_resolution=self.input_resolution[::-1]
        # imgs = F.interpolate(imgs, size=self.input_resolution,
        #                      mode='bilinear', align_corners=False)
        # ir = F.interpolate(ir, size=self.input_resolution,
        #                      mode='bilinear', align_corners=False)
        if self.share_conv:
            imgs,ir=self.head(imgs),self.head(ir)
        else:
            imgs, ir = self.head(imgs), self.headText(ir)

        # _,_,H,W=imgs.shape
        # ir = F.interpolate(ir, size=(H,W),
        #                    mode='bilinear', align_corners=False)
        # imgs=self.body(imgs,ir)
        # imgs=self.re(imgs)
        # imgs=self.tail(imgs)
        imgs=self.tail2(imgs)
        imgs=self.liner(imgs.squeeze(3).squeeze(2))
        return imgs

def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        net.fc = nn.Linear(512, out_dim)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f


##############################
#        Discriminator
##############################


def discriminator_block(in_filters, out_filters, normalization=False,stride=2):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))

    return layers


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128),
            nn.Conv2d(128, 1, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)

class IrRefine(nn.Module):
    def __init__(self, embed_dim, fusion=True,share_conv=True):
        super(IrRefine, self).__init__()
        self.embed_dim = embed_dim
        self.fusion = fusion
        self.share_conv=share_conv
        if self.fusion:
            self.pre_conv = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding=2,dilation=2)
            self.pre_conv1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding=2, dilation=2)
            self.post_conv = BasicBlock(embed_dim * 4, embed_dim,kernel_size=1,stride=1,padding=0,norm=True )
        else:
            self.group_conv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.Conv2d(embed_dim, embed_dim * 2, 7, 1, 3, groups=embed_dim)
            )

            self.post_conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x, inf):
        _,_,W,H=x.shape
        inf = F.interpolate(inf, size=(W,H),
                           mode='bilinear', align_corners=False)
        if self.fusion:
            if self.share_conv:
                x0, x1 = self.pre_conv(x), self.pre_conv(inf)
            else:
                x0, x1 = self.pre_conv(x), self.pre_conv1(inf)
            # x_ = F.gelu(x0) * torch.sigmoid(x1)
            x_=self.post_conv(torch.cat([x0,x1],dim=1))
        else:
            B, C, H, W = x.size()
            x0, x1 = self.group_conv(x).view(B, C, 2, H, W).chunk(2, dim=2)
            x_ = F.gelu(x0.squeeze(2)) * torch.sigmoid(x1.squeeze(2))
            x_ = self.post_conv(x_)
        return x_
class BasicBlock(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,norm=False,groups=1,dilation=1):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,groups=groups,dilation=dilation),
            nn.LeakyReLU(0.05)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class Classifier(nn.Module):
    def __init__(self, in_channels=3, input_resolution=256, extra_pooling=False):
        super(Classifier, self).__init__()
        self.input_resolution = input_resolution
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )
        self.model2 = nn.Sequential(
            BasicBlock(3, 16, stride=2, norm=True),
            BasicBlock(16, 32, stride=2, norm=True),
            # BasicBlock(32, 64, stride=2, norm=True),
            # BasicBlock(64, 128, stride=2, norm=True),
            # BasicBlock(128, 128, stride=2),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(2),
            BasicBlock(32, 32, stride=2),
        )
        self.liner = nn.Sequential(
            nn.Linear(32, 3),
        )

    def forward(self, img_input,ir):
        img_input = F.interpolate(img_input, size=(self.input_resolution,) * 2,
                                  mode='bilinear', align_corners=False)
        imgs = self.model2(img_input).squeeze(3).squeeze(2)
        imgs = self.liner(imgs)
        return imgs


class Classifier_unpaired(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier_unpaired, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)

class Generator3DLUT_identity(nn.Module):
    def __init__(self, device,dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()
        self.LUT.to(device)
        self.TrilinearInterpolation.to(device)

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        # self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output



class Generator3DLUT_zero(nn.Module):
    def __init__(self,device, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        self.LUT = nn.Parameter(self.LUT.clone().detach().requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

        self.LUT.to(device)
        self.TrilinearInterpolation.to(device)

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output



class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        assert 1 == trilinear.forward(lut,
                                      x,
                                      output,
                                      dim,
                                      shift,
                                      binsize,
                                      W,
                                      H,
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_tensors
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        assert 1 == trilinear.backward(x,
                                       x_grad,
                                       lut_grad,
                                       dim,
                                       shift,
                                       binsize,
                                       W,
                                       H,
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module):
    def __init__(self,device, dim=33 ):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float,device=device)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float,device=device)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float,device=device)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-uncased', mirror='tuna')

        self.conv = nn.Sequential(
            Rearrange("B (C W H) -> B C W H", C=3, W=16, H=16),
        )


    def forward(self, input_ids):
        # outputs = self.bert(input_ids, attention_mask=attention_mask)
        # pooled_output = outputs[1]
        return self.conv(input_ids)

class IRLUT(nn.Module):
    def __init__(self, device,n_lut=3,lut_dim=33,press_radio=8,ch_radio=8,expan_att_chans=4,lead_strady="cat",reFine=True,lut_share=True,att_share=True,refine_share=True ):
        super(IRLUT, self).__init__()
        self.LUT = nn.ModuleList()
        self.n_lut = n_lut
        self.clamp = True
        self.LUT.append(Generator3DLUT_identity(device=device).to(device))
        for i in range(n_lut-1):
            self.LUT.append(Generator3DLUT_zero(device=device).to(device))
        self.classifier =IRBackbone(n_lut=n_lut,device=device,press_radio=press_radio,ch_radio=ch_radio,lead_strady=lead_strady,expan_att_chans=expan_att_chans,share_conv=lut_share,att_share=att_share)
        # self.classifier =Classifier()
        self.classifier.to(device)
        self.textEmbding=TextEncoder()
        self.textEmbding.to(device)
        self.reFine=reFine
        if reFine:
            self.refine=IrRefine(3,refine_share)
            self.refine.to(device)
        self.TV3 = TV_3D(dim=lut_dim, device=device).to(device)
        self.trilinear_ = TrilinearInterpolation().to(device)
        self.classifier.eval()

    def getTvMn(self):
        tv_mn = [self.TV3(lut) for lut in self.LUT]
        tv_cons = sum(tv for tv, _ in tv_mn)
        mn_cons = sum(mn for _, mn in tv_mn)
        return tv_cons, mn_cons

    def forward(self, img, text, inf):

        pred = self.classifier(img, inf).squeeze()
        LUT = sum(pred[i] * self.LUT[i].LUT for i in range(self.n_lut))

        weights_norm = torch.mean(pred ** 2)

        # combine_A = img.new(img.size())
        _, combine_A = self.trilinear_(LUT, img)
        if self.reFine:
            _, _, W, H = img.shape
            inf = F.interpolate(inf, size=(W, H), mode='bilinear', align_corners=False)
            refineImg = self.refine(img, inf) + combine_A.detach()
            return combine_A, weights_norm, refineImg
        else:
            return combine_A, weights_norm, combine_A

    def forward_train(self, img, text, inf):
        if modelCal:
            img=img[0,:,:,:].unsqueeze(0)
            inf=inf[0,:,:,:].unsqueeze(0)
        pred = self.classifier(img, inf).squeeze()
        # LUT = sum(pred[i] * self.LUT[i].LUT for i in range(self.n_lut))

        weights_norm = torch.mean(pred ** 2)

        # combine_A = img.new(img.size())
        # _, combine_A = self.trilinear_(LUT, img)
        # combine_A = img.new(img.size())
        combine_A = sum(pred[i] * self.LUT[i](img) for i in range(self.n_lut))
        if self.reFine:
            _, _, W, H = img.shape
            inf = F.interpolate(inf, size=(W, H), mode='bilinear', align_corners=False)
            refineImg = self.refine(img, inf) + combine_A.detach()
            return combine_A, weights_norm, refineImg
        else:
            return combine_A, weights_norm, combine_A




class Enhancer():
    def __init__(self, config):
        super(Enhancer, self).__init__()
        self.model = IRLUT(lut_dim=config.lut_dim,
                           n_lut=config.lut_n,
                           device=config.device,
                           press_radio=config.lut.press_radio,
                           lead_strady=config.lut.lead_strady,
                           ch_radio=config.lut.ch_radio,
                           expan_att_chans=config.expan_att_chans,
                           reFine=config.isRefine,
                           lut_share=config.lut.lut_share,
                           refine_share=config.refine_share)
        self.model.to(config.device)
        self.config = config
        self.criterion = torch.nn.MSELoss()
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=0.01)

        self.criterion_pixelwise = torch.nn.MSELoss()
        # lut_parameters = [lut.parameters() for lut in self.model.LUT]
        # all_parameters = itertools.chain(self.model.parameters(), *lut_parameters)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(config.b1, config.b2),eps=1e-8,weight_decay=0)

        # self.optimizer = torch.optim.Adam(
        #     itertools.chain(self.model.parameters(), self.model.LUT[0].parameters(), self.model.LUT[1].parameters(),
        #                     self.model.LUT[2].parameters()),
        #     lr=config.lr, betas=(config.b1, config.b2))  # , LUT3.parameters(), LUT4.parameters()

        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)
    def train_on_batch(self, batch: list):
        inputs = batch["A_input"].to(self.config.device)
        labels = batch["A_exptC"].to(self.config.device)
        inf = batch["A_Inf"].to(self.config.device)
        text = batch["img_text"].to(self.config.device)
        outputs, weights_norm,refineImg = self.model(inputs.detach(), text.detach(), inf.detach())
        tv_cons, mn_cons = self.model.getTvMn()
        mse=self.criterion(outputs, labels.detach())
        mse2=self.criterion_pixelwise(refineImg,labels.detach())

        loss = self.config.ir_weight*mse2+ self.config.lut_weight*mse+ self.config.lambda_smooth * (
                    weights_norm + tv_cons) +  self.config.lambda_monotonicity * mn_cons
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        psnr =10 * math.log10(1 / mse2.item())
        ssim = kornia.metrics.ssim(
            ((refineImg + 1) / 2).clamp(0, 1),
            ((labels.detach() + 1) / 2).clamp(0, 1),
            window_size=11,
        ).mean()
        meter = {
            'loss': loss.item(),
            'psnr': psnr,
            'ssim': ssim.item()
        }
        return meter, (inputs, labels, outputs, inf, text)

    # def sche_step(self):
    #     self.scheduler.step()

    def validate_on_batch(self, batch: list):
        inputs = batch["A_input"].to(self.config.device)
        labels = batch["A_exptC"].to(self.config.device)
        inf = batch["A_Inf"].to(self.config.device)
        text = batch["img_text"].to(self.config.device)

        with torch.no_grad():
            outputs, weights_norm,refineImg = self.model(inputs.detach(), text.detach(), inf.detach())
        tv_cons, mn_cons = self.model.getTvMn()
        _,_,W,H=outputs.shape
        labels=F.interpolate(labels,(W,H),mode='bilinear', align_corners=False)
        mse = self.criterion(outputs, labels.detach())
        mse2 = self.criterion_pixelwise(refineImg, labels.detach())
        loss =  self.config.ir_weight*mse2+ self.config.lut_weight*mse+ self.config.lambda_smooth * (
                    weights_norm + tv_cons) +  self.config.lambda_monotonicity * mn_cons

        psnr = 10 * math.log10(1 / mse2.item())
        ssim = kornia.metrics.ssim(
            ((refineImg + 1) / 2).clamp(0, 1),
            ((labels.detach() + 1) / 2).clamp(0, 1),
            window_size=11,
        ).mean()
        meter = {
            'loss': loss.item(),
            'psnr': psnr,
            'ssim': ssim.item()
        }
        return meter, (inputs, labels, outputs, inf, text)
import torchsummary as summary
def allModel():
    args = OmegaConf.load("option/train_fivek.yaml")
    config = args.train
    config.device='cuda:0'
    torch.cuda.set_device(config.device)
    model = IRLUT(lut_dim=config.lut_dim,
                       n_lut=config.lut_n,
                       device=config.device,
                       press_radio=config.lut.press_radio,
                       lead_strady=config.lut.lead_strady,
                       ch_radio=config.lut.ch_radio,
                       expan_att_chans=config.expan_att_chans,
                       reFine=config.isRefine,
                       lut_share=config.lut.lut_share,
                       refine_share=config.refine_share)
    summary.summary(model, [(3, 480, 720), (1, 768), (3,480, 720)])
def generate_LUT(classifier,img,):
    pred = classifier(img).squeeze()

    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    return LUT
def fps():
    args = OmegaConf.load("option/train_fivek.yaml")
    config = args.train
    config.device = 'cuda:0'
    torch.cuda.set_device(config.device)

    model = IRLUT(lut_dim=config.lut_dim,
                  n_lut=config.lut_n,
                  device=config.device,
                  press_radio=config.lut.press_radio,
                  lead_strady=config.lut.lead_strady,
                  ch_radio=config.lut.ch_radio,
                  expan_att_chans=config.expan_att_chans,
                  reFine=config.isRefine,
                  lut_share=config.lut.lut_share,
                  refine_share=config.refine_share)
    # self.classifier =Classifier()
    classifier = model.cuda()

    classifier.eval()


    res = []
    img= torch.empty(1, 3, 480, 720).cuda()
    inf = torch.empty(1, 3, 480, 720).cuda()

    for id in enumerate(range(0,200)):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            classifier(img,None,inf)  # 有待修改
        torch.cuda.synchronize()
        end = time.time()
        res.append(end - start)
    time_sum = 0
    for i in res:
        time_sum += i
    print("FPS: %f" % (1.0 / (time_sum / len(res))))
    print("time: %f ms" % ( time_sum / len(res)))

if __name__ == '__main__':
    modelCal=True
    allModel()
    modelCal = False
    fps()