_base_ = [
    '../../_base_/datasets/mmseg/cityscapes.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_40k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

# model settings
# deeplabv3 r18
student = dict(
    type='mmseg.EncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'  # noqa: E501

# pspnet r101
teacher = dict(
    type='mmseg.EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

alpha_mgd=0.00002
lambda_mgd=0.75
# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMSegArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='backbone.layer4',
                teacher_module='backbone.layer4',
                losses=[
                    dict(
                        type='MGDLoss',
                        name='loss_mgd',
                        alpha_mgd=alpha_mgd,
                        lambda_mgd=lambda_mgd,
                        # teacher_channels=[2048, 2048, 2048, 2048, 2048],
                    ),
                ],
                align_module=dict(
                    type='conv2d',
                    num_modules=1,
                    student_channels=512,
                    teacher_channels=2048,
                ))
        ]),
)

find_unused_parameters = True