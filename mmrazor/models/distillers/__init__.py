# Copyright (c) OpenMMLab. All rights reserved.
from .self_distiller import SelfDistiller
from .single_teacher_ORI import SingleTeacherDistiller

__all__ = ['SelfDistiller', 'SingleTeacherDistiller']
