from typing import Any, Dict, Type, Union

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
)
from autoPyTorch.pipeline.components.setup.network.backbone.base_backbone import BaseBackbone
from autoPyTorch.pipeline.components.setup.network.backbone.image import ConvNetImageBackbone, DenseNetImageBackbone, \
    ResNetImageBackbone
from autoPyTorch.pipeline.components.setup.network.backbone.tabular import MLPBackbone, ResNetBackbone, \
    ShapedMLPBackbone, ShapedResNetBackbone
from autoPyTorch.pipeline.components.setup.network.backbone.time_series import InceptionTimeBackbone, TCNBackbone

_backbones = {
    ConvNetImageBackbone.get_name(): ConvNetImageBackbone,
    DenseNetImageBackbone.get_name(): DenseNetImageBackbone,
    ResNetImageBackbone.get_name(): ResNetImageBackbone,
    ResNetBackbone.get_name(): ResNetBackbone,
    ShapedResNetBackbone.get_name(): ShapedResNetBackbone,
    ShapedMLPBackbone.get_name(): ShapedMLPBackbone,
    MLPBackbone.get_name(): MLPBackbone,
    TCNBackbone.get_name(): TCNBackbone,
    InceptionTimeBackbone.get_name(): InceptionTimeBackbone
}
_addons = ThirdPartyComponents(BaseBackbone)


def add_backbone(backbone: BaseBackbone) -> None:
    _addons.add_component(backbone)


def get_available_backbones() -> Dict[str, Union[Type[BaseBackbone], Any]]:
    backbones = dict()
    backbones.update(_backbones)
    backbones.update(_addons.components)
    return backbones
