_base_ = [
    '../../_base_/datasets/mmseg/cityscapes.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_40k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

# pspnet r18
student = dict(
    type='mmseg.EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'),
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
        type='PSPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pool_scales=(1, 2, 3, 6),
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
                        type='ChannelWiseDivergence',
                        name='loss_cwd',
                        tau=1,
                        loss_weight=5,
                    )
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