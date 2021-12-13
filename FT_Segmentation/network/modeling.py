from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet


def _segm_resnet(name, backbone_name, input_channels, num_classes, output_stride, pretrained_backbone):

    
    replace_stride_with_dilation=[False, False, True]
    aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](input_channels=input_channels,
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    # print('backbone',backbone)
    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(arch_type, backbone, input_channels, num_classes, output_stride, pretrained_backbone):

    model = _segm_resnet(arch_type, backbone, input_channels=input_channels, num_classes=num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    return model

def deeplabv3plus_resnet101(input_channels, num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    
    return _load_model('deeplabv3plus', 'resnet101', input_channels=input_channels, num_classes=num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
