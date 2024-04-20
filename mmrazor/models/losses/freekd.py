import math
import torch
import torch.nn as nn
from mmcv.runner.checkpoint import _load_checkpoint
import cv2
from ..builder import LOSSES

from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np

def dice_coeff(inputs):
    # inputs: [B, T, H*W]
    pred = inputs[:, None, :, :]
    target = inputs[:, :, None, :]

    mask = pred.new_ones(pred.size(0), target.size(1), pred.size(2))
    mask[:, torch.arange(mask.size(1)), torch.arange(mask.size(2))] = 0

    a = torch.sum(pred * target, -1)
    b = torch.sum(pred * pred, -1) + 1e-12
    c = torch.sum(target * target, -1) + 1e-12
    d = (2 * a) / (b + c)
    d = (d * mask).sum() / mask.sum()
    return d

class MaskModule(nn.Module):
    def __init__(self, channels, num_tokens=6, weight_mask=True):
        super().__init__()
        self.weight_mask = weight_mask
        self.mask_token = nn.Parameter(torch.randn(4, num_tokens, channels).normal_(0, 0.01)) # 2, T, C
        if self.weight_mask:
            self.prob = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, num_tokens, kernel_size=1)
            )
            self.init_weights()

        self.xfm = DWTForward(J=3, mode='zero',wave='haar')
        self.ixfm = DWTInverse(mode='zero',wave='haar')

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_mask(self, x, mask_token):
        N, C, H, W = x.shape
        mask_token = mask_token.expand(N, -1, -1)  # [N, T, C]
        k = x.view(N, -1, H * W)
        attn = mask_token @ k  # [N, T, H * W]
        attn = attn.sigmoid()
        attn = attn.view(N, -1, H, W)
        return attn

    def forward_prob(self, x):
        mask_probs = self.prob(x)  # [N, T, 1, 1]
        mask_probs = mask_probs.softmax(1).unsqueeze(2)  # [N, T, 1, 1, 1]
        return mask_probs

    def forward_train(self, x):
        # Split
        cA, (cH, cV, cD) = self.xfm(x)

        slice_cH = cH.sum(2) # [N, C, H, W]
        slice_cV = cV.sum(2) # [N, C, H, W]
        slice_cD = cD.sum(2) # [N, C, H, W]
        # print("test", slice_cH.shape, slice_cV.shape, slice_cD.shape)

        # Low Frequnecy [N, C, H, W]
        mask_cA = self.forward_mask(cA, self.mask_token[0])
        out_cA = cA.unsqueeze(1) * mask_cA.unsqueeze(2)  # [N, T, C, H, W]
        out_cA = out_cA.sum(1) # [N, C, H, W]

        # cH
        mask_cH = self.forward_mask(slice_cH, self.mask_token[1])
        out_cH = cH.unsqueeze(1) * mask_cH.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]
        out_cH = out_cH.sum(1)

        # cV
        mask_cV = self.forward_mask(slice_cV, self.mask_token[2])
        out_cV = cV.unsqueeze(1) * mask_cV.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]
        out_cV = out_cV.sum(1)

        # High Frequnecy [N, C, H, W]
        mask_cD = self.forward_mask(slice_cD, self.mask_token[3])
        out_cD = cD.unsqueeze(1) * mask_cD.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]
        out_cD = out_cD.sum(1)

        # loss
        mask_loss = dice_coeff(mask_cA.flatten(2)) + dice_coeff(mask_cD.flatten(2)) \
        + dice_coeff(mask_cV.flatten(2)) + dice_coeff(mask_cH.flatten(2))

        # merge
        out = self.ixfm((out_cA, (out_cH, out_cV, out_cD)))

        return out, mask_loss

    def forward(self, x):
        return self.forward_train(x)


@LOSSES.register_module()
class FreeKDLoss(nn.Module):

    def __init__(self, channels, num_tokens=2, weight_mask=True, custom_mask=True, custom_mask_warmup=1000, pretrained=None, loss_weight=1.):
        super().__init__()
        self.loss_weight = loss_weight
        self.weight_mask = weight_mask
        self.custom_mask = custom_mask
        self.custom_mask_warmup = custom_mask_warmup

        self.mask_modules = nn.ModuleList([
            MaskModule(channels=c, num_tokens=num_tokens, weight_mask=weight_mask) for c in channels]
        )

        self.init_weights(pretrained)

        self.xfm = DWTForward(J=3, mode='zero',wave='haar')
        self.ixfm = DWTInverse(mode='zero',wave='haar')
        self._iter = 0

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return
        ckpt = _load_checkpoint(pretrained, map_location='cpu')
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if 'mask_modules' in k:
                state_dict[k[k.find('mask_modules'):]] = v
        self.load_state_dict(state_dict, strict=True)

    def forward(self, y_s_list, y_t_list):
        if not isinstance(y_s_list, (tuple, list)):
            y_s_list = (y_s_list, )
            y_t_list = (y_t_list, )
        assert len(y_s_list) == len(y_t_list) == len(self.mask_modules)

        losses = []
        for stage, (y_s, y_t, mask_module) in enumerate(zip(y_s_list, y_t_list, self.mask_modules)):
            # Split
            cA, (cH, cV, cD) = self.xfm(y_t)
            cA_s, (cH_s, cV_s, cD_s) = self.xfm(y_s)

            # predict the masks
            slice_cH = cH.sum(2) # [N, C, H, W]
            slice_cV = cV.sum(2) # [N, C, H, W]
            slice_cD = cD.sum(2) # [N, C, H, W]

            # Low Frequnecy [N, C, H, W]
            # mask_cA = mask_module.forward_mask(cA, mask_module.mask_token[0])
            # out_cA = cA.unsqueeze(1) * mask_cA.unsqueeze(2)  # [N, T, C, H, W]
            # out_cA = out_cA.sum(1) # [N, C, H, W]

            # cH
            mask_cH = mask_module.forward_mask(slice_cH, mask_module.mask_token[1])
            out_cH = cH.unsqueeze(1) * mask_cH.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]
            out_cH_s = cH_s.unsqueeze(1) * mask_cH.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]

            # cV
            mask_cV = mask_module.forward_mask(slice_cV, mask_module.mask_token[2])
            out_cV = cV.unsqueeze(1) * mask_cV.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]
            out_cV_s = cV_s.unsqueeze(1) * mask_cV.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]

            # High Frequnecy [N, C, H, W]
            mask_cD = mask_module.forward_mask(slice_cD, mask_module.mask_token[3])
            out_cD = cD.unsqueeze(1) * mask_cD.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]
            out_cD_s = cD_s.unsqueeze(1) * mask_cD.unsqueeze(2).unsqueeze(2) # [N, T, C, J, H, W]

            # masked distillation
            loss = torch.nn.functional.l1_loss(out_cH_s, out_cH) + \
            torch.nn.functional.l1_loss(out_cV_s, out_cV) + \
            torch.nn.functional.l1_loss(out_cD_s, out_cD)
            losses.append(loss)
        
        loss = sum(losses)
        self._iter += 1
        return self.loss_weight * loss