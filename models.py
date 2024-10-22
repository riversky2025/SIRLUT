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


class ChannelAttentionRs(nn.Module):
    def __init__(self, embed_dim, num_chans=2, expan_att_chans=1, ):
        super(ChannelAttentionRs, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.ch_radio = 1
        self.qkvexten = 2

        self.t = nn.Parameter(torch.ones(1, self.num_heads * self.ch_radio, 1, 1))
        self.t2 = nn.Parameter(torch.ones(1, self.num_heads * self.ch_radio, 1, 1))

        self.group_qkv_rgb = nn.Sequential(
            BasicBlock(embed_dim, embed_dim * expan_att_chans * 3, kernel_size=3, groups=embed_dim, norm=True),
            Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * 3),
        )

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

        _, kir, vir = self.group_qkv_rgb(ir).contiguous().chunk(3, dim=1)

        kir = kir.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        vir = vir.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        x_ = self.attnfun(qv, kv, vv, self.t)
        x_ = self.attnfun(x_, kir, vir, self.t2)
        x_ = rearrange(x_, "B C X (H W)-> B (X C) H W", B=B, W=W, H=H).contiguous()
        x_ = self.group_fus(x_)
        return x_


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 ch_radio,
                 press_radio,
                 expan_att_chans
                 ):
        super(TransformerBlock, self).__init__()
        ya, ch = press_radio, ch_radio
        ch_input = int(embed_dim * ch * ch / ya)
        self.norm1 = nn.ModuleList([
            nn.Sequential(
                Rearrange('B C H W -> B (H W) C'),
                nn.LayerNorm(embed_dim)
            ) for _ in range(2)
        ])
        self.embed_dim = embed_dim
        self.headPre = nn.ModuleList([
            nn.Sequential(
                nn.PixelUnshuffle(ch_radio),
                BasicBlock(embed_dim * ch_radio * ch_radio, ch_input, norm=True),
            ) for _ in range(2)
        ])
        self.body = ChannelAttentionRs(ch_input, expan_att_chans=expan_att_chans)
        self.norm2 = nn.ModuleList([
            nn.Sequential(
                Rearrange('B C H W -> B (H W) C'),
                nn.LayerNorm(embed_dim)
            ) for _ in range(2)
        ])

    def forward(self, batch):
        x, inf = batch
        B, C, H, W = x.size()
        x_ = rearrange(self.norm1[0](x), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x_inf = rearrange(self.norm2[1](inf), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x_ = self.headPre[0](x_)
        x_inf = self.headPre[1](x_inf)
        x_ = self.body(x_, x_inf)
        return x_, x_inf


class IRBackbone(nn.Module):
    # (360,540) (480,720) 448 448
    def __init__(self, n_lut=3, input_resolution=(448, 720), expan_att_chans=4):
        super().__init__()

        dim = 3
        embed_dim = 16
        self.pred_embed = nn.Conv2d(dim, embed_dim, 3, 1, 1)
        self.pred_ir_embed = nn.Conv2d(dim, embed_dim, 3, 1, 1)
        num_blocks = (1, 1)
        self.encoder = nn.ModuleList([nn.Sequential(*[
            TransformerBlock(
                embed_dim * 2 ** i, 2,
                2, expan_att_chans) for _ in range(num_blocks[i])
        ]) for i in range(len(num_blocks))])

        self.downsampler = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(embed_dim * 2 ** i), int(embed_dim * 2 ** (i - 1)), 3, 1, 1),
                nn.PixelUnshuffle(2)
            ) for i in range(len(num_blocks) - 2)
        ]).append(nn.Identity())

        ch_input = 32
        self.tail2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(2),
            BasicBlock(ch_input, ch_input, stride=2),
        )
        self.liner = nn.Sequential(
            nn.Linear(ch_input, n_lut),
        )
        self.input_resolution = input_resolution

    def forward(self, imgs, ir):
        _, _, W, H = imgs.shape
        self.input_resolution = tuple(sorted(self.input_resolution))
        if W > H:
            self.input_resolution = self.input_resolution[::-1]
        imgs = F.interpolate(imgs, size=self.input_resolution,
                             mode='bilinear', align_corners=False)
        ir = F.interpolate(ir, size=self.input_resolution,
                           mode='bilinear', align_corners=False)
        imgs = self.pred_embed(imgs)
        ir = self.pred_ir_embed(ir)
        for layer, sampler,  in zip(
                self.encoder, self.downsampler
        ):
            imgs, _ = layer((imgs, ir))
            ir = sampler(ir)
        imgs = self.tail2(imgs)
        imgs = self.liner(imgs.squeeze(3).squeeze(2))
        return imgs


class IrRefine(nn.Module):
    def __init__(self, embed_dim):
        super(IrRefine, self).__init__()
        self.embed_dim = embed_dim
        self.pre_conv = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.post_conv = BasicBlock(embed_dim * 4, embed_dim, kernel_size=1, stride=1, padding=0, norm=True)

    def forward(self, x, inf):
        _, _, W, H = x.shape
        inf = F.interpolate(inf, size=(W, H),
                            mode='bilinear', align_corners=False)

        x0, x1 = self.pre_conv(x), self.pre_conv(inf)

        x_ = self.post_conv(torch.cat([x0, x1], dim=1))
        return x_


class BasicBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=False, groups=1, dilation=1):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                      dilation=dilation),
            nn.LeakyReLU(0.05)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class Generator3DLUT_identity(nn.Module):
    def __init__(self, device, dim=33):
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
    def __init__(self, device, dim=33):
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
    def __init__(self, device, dim=33):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float, device=device)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float, device=device)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float, device=device)
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


class IRLUT(nn.Module):
    def __init__(self, device, n_lut=3, lut_dim=33, press_radio=8, ch_radio=8, expan_att_chans=4, lead_strady="cat",
                 reFine=True, lut_share=True, att_share=True, refine_share=True):
        super(IRLUT, self).__init__()
        self.LUT = nn.ModuleList()
        self.n_lut = n_lut
        self.clamp = True
        self.LUT.append(Generator3DLUT_identity(device=device).to(device))
        for i in range(n_lut - 1):
            self.LUT.append(Generator3DLUT_zero(device=device).to(device))
        self.classifier = IRBackbone(n_lut=n_lut, expan_att_chans=expan_att_chans)
        # self.classifier =Classifier()
        self.classifier.to(device)

        self.refine = IrRefine(3)
        self.refine.to(device)
        self.TV3 = TV_3D(dim=lut_dim, device=device).to(device)
        self.trilinear_ = TrilinearInterpolation().to(device)

    def getTvMn(self):
        tv_mn = [self.TV3(lut) for lut in self.LUT]
        tv_cons = sum(tv for tv, _ in tv_mn)
        mn_cons = sum(mn for _, mn in tv_mn)
        return tv_cons, mn_cons

    def forward(self, img, inf, train=True):

        imgRef = img.clone()
        infRef = inf.clone()
        pred = self.classifier(img, inf)
        pred = pred.squeeze()
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(0)
        if train:
            combine_A = img.new(img.size())
            for b in range(img.size(0)):
                LUTIMG = []
                for i in range(self.n_lut):
                    LUTIMG.append(self.LUT[i](img))
                combine_A[b, :, :, :] = sum(pred[b, i] * LUTIMG[i][b, :, :, :] for i in range(self.n_lut))
        else:
            LUT = sum(pred[0][i] * self.LUT[i].LUT for i in range(self.n_lut))
            _, combine_A = self.trilinear_(LUT, img)
        weights_norm = torch.mean(pred ** 2)
        _, _, W, H = img.shape
        infRef = F.interpolate(infRef, size=(W, H), mode='bilinear', align_corners=False)
        refineImg = self.refine(imgRef.detach(), infRef.detach()) + combine_A.detach()
        return combine_A, weights_norm, refineImg

    def forward_train(self, img, text, inf):
        if modelCal:
            img = img[0, :, :, :].unsqueeze(0)
            inf = inf[0, :, :, :].unsqueeze(0)
        pred = self.classifier(img, inf).squeeze()

        weights_norm = torch.mean(pred ** 2)
        combine_A = sum(pred[i] * self.LUT[i](img) for i in range(self.n_lut))
        _, _, W, H = img.shape
        inf = F.interpolate(inf, size=(W, H), mode='bilinear', align_corners=False)
        refineImg = self.refine(img, inf) + combine_A.detach()
        return combine_A, weights_norm, refineImg


class Enhancer():
    def __init__(self, config):
        super(Enhancer, self).__init__()
        self.model = IRLUT(lut_dim=config.lut_dim,
                           n_lut=config.lut_n,
                           device=config.device,
                           press_radio=config.lut.press_radio,
                           ch_radio=config.lut.ch_radio,
                           expan_att_chans=config.expan_att_chans,
                           )
        self.model.to(config.device)
        self.config = config
        self.criterion = torch.nn.MSELoss()

        self.criterion_pixelwise = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(config.b1, config.b2), eps=1e-8,
                                          weight_decay=0)


    def train_on_batch(self, batch: list):
        inputs = batch["A_input"].to(self.config.device)
        labels = batch["A_exptC"].to(self.config.device)

        inf = batch["A_Inf"].to(self.config.device)
        outputs, weights_norm, refineImg = self.model(inputs.detach(), inf.detach())
        tv_cons, mn_cons = self.model.getTvMn()
        mse = self.criterion(outputs, labels.detach())
        mse2 = self.criterion_pixelwise(refineImg, labels.detach())

        loss = self.config.ir_weight * mse2 + self.config.lut_weight * mse + self.config.lambda_smooth * (
                weights_norm + tv_cons) + self.config.lambda_monotonicity * mn_cons
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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
        return meter, (inputs, labels, outputs, inf, inf)

    def validate_on_batch(self, batch: list):
        inputs = batch["A_input"].to(self.config.device)
        labels = batch["A_exptC"].to(self.config.device)

        inf = batch["A_Inf"].to(self.config.device)
        with torch.no_grad():
            outputs, weights_norm, refineImg = self.model(inputs.detach(), inf.detach(), False)
        tv_cons, mn_cons = self.model.getTvMn()
        _, _, W, H = outputs.shape
        labels = F.interpolate(labels, (W, H), mode='bilinear', align_corners=False)
        mse = self.criterion(outputs, labels.detach())
        mse2 = self.criterion_pixelwise(refineImg, labels.detach())
        loss = self.config.ir_weight * mse2 + self.config.lut_weight * mse + self.config.lambda_smooth * (
                weights_norm + tv_cons) + self.config.lambda_monotonicity * mn_cons

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
        return meter, (inputs, labels, outputs, inf, inf)
