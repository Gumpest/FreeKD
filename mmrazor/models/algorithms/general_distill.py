# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.builder import ALGORITHMS
from mmrazor.models.utils import add_prefix
from .base import BaseAlgorithm

import torch
import torch.nn as nn

class FixedPatchPrompter_image(nn.Module):
    def __init__(self, prompt_size):
        super(FixedPatchPrompter_image, self).__init__()
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch
        return x + prompt


@ALGORITHMS.register_module()
class GeneralDistill(BaseAlgorithm):
    """General Distillation Algorithm.

    Args:
        with_student_loss (bool): Whether to use student loss.
            Defaults to True.
        with_teacher_loss (bool): Whether to use teacher loss.
            Defaults to False.
    """

    def __init__(self,
                 with_student_loss=True,
                 with_teacher_loss=False,
                 **kwargs):

        super(GeneralDistill, self).__init__(**kwargs)
        self.with_student_loss = with_student_loss
        self.with_teacher_loss = with_teacher_loss

        # self.visual_prompt = FixedPatchPrompter_image(prompt_size=100)

    def train_step(self, data, optimizer):
        """"""
        
        # print("test", data['img'].shape) # ['img_metas', 'img', 'gt_bboxes', 'gt_labels']

        losses = dict()
        if self.with_student_loss:
            # print("test1", data['img'][:, :, 0, 0])
            # data['img'] = self.visual_prompt(data['img'])
            # print("test2", data['img'][:, :, 0, 0])

            # print(self.visual_prompt.patch.data.requires_grad_())
            student_losses = self.distiller.exec_student_forward(
                self.architecture, data)
            student_losses = add_prefix(student_losses, 'student')
            losses.update(student_losses)
        else:
            # Just to be able to trigger the forward hooks that
            # have been registered
            _ = self.distiller.exec_student_forward(self.architecture, data)

        if self.with_teacher_loss:
            teacher_losses = self.distiller.exec_teacher_forward(data)
            teacher_losses = add_prefix(teacher_losses, 'teacher')
            losses.update(teacher_losses)
        else:
            # Just to be able to trigger the forward hooks that
            # have been registered
            _ = self.distiller.exec_teacher_forward(data)

        distill_losses = self.distiller.compute_distill_loss(data)
        distill_losses = add_prefix(distill_losses, 'distiller')
        losses.update(distill_losses)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        return outputs


