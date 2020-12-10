import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import ConfigSpace as CS
import torch
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)
from torch import nn
from torch.nn import functional as F

from autoPyTorch.pipeline.components.setup.network.backbone.base_backbone import BaseBackbone
from autoPyTorch.pipeline.components.setup.network.utils.common import initialize_weights
from autoPyTorch.pipeline.components.setup.network.utils.shake_image import generate_alpha_beta_single, shake_drop, \
    generate_alpha_beta, shake_shake

_activations: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}


class ConvNetImageBackbone(BaseBackbone):
    supported_tasks = {"image_classification", "image_regression"}

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.bn_args = {"eps": 1e-5, "momentum": 0.1}

    def _get_layer_size(self, w: int, h: int) -> Tuple[int, int]:
        cw = ((w - self.config["conv_kernel_size"] + 2 * self.config["conv_kernel_padding"])
              // self.config["conv_kernel_stride"]) + 1
        ch = ((h - self.config["conv_kernel_size"] + 2 * self.config["conv_kernel_padding"])
              // self.config["conv_kernel_stride"]) + 1
        cw, ch = cw // self.config["pool_size"], ch // self.config["pool_size"]
        return cw, ch

    def _add_layer(self, layers: List[nn.Module], in_filters: int, out_filters: int) -> None:
        layers.append(nn.Conv2d(in_filters, out_filters,
                                kernel_size=self.config["conv_kernel_size"],
                                stride=self.config["conv_kernel_stride"],
                                padding=self.config["conv_kernel_padding"]))
        layers.append(nn.BatchNorm2d(out_filters, **self.bn_args))
        layers.append(_activations[self.config["activation"]]())
        layers.append(nn.MaxPool2d(kernel_size=self.config["pool_size"], stride=self.config["pool_size"]))

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        channels, iw, ih = input_shape
        layers: List[nn.Module] = []
        init_filter = self.config["conv_init_filters"]
        self._add_layer(layers, channels, init_filter)

        cw, ch = self._get_layer_size(iw, ih)
        for i in range(2, self.config["num_layers"] + 1):
            cw, ch = self._get_layer_size(cw, ch)
            if cw == 0 or ch == 0:
                logging.info("> reduce network size due to too small layers.")
                break
            self._add_layer(layers, init_filter, init_filter * 2)
            init_filter *= 2
        backbone = nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'ConvNetImageBackbone',
            'name': 'ConvNetImageBackbone',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_num_layers: int = 2,
                                        max_num_layers: int = 5,
                                        min_init_filters: int = 16,
                                        max_init_filters: int = 64,
                                        min_kernel_size: int = 2,
                                        max_kernel_size: int = 5,
                                        min_stride: int = 1,
                                        max_stride: int = 3,
                                        min_padding: int = 2,
                                        max_padding: int = 3,
                                        min_pool_size: int = 2,
                                        max_pool_size: int = 3) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(UniformIntegerHyperparameter('num_layers',
                                                           lower=min_num_layers,
                                                           upper=max_num_layers))
        cs.add_hyperparameter(CategoricalHyperparameter('activation',
                                                        choices=list(_activations.keys())))
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_init_filters',
                                                           lower=min_init_filters,
                                                           upper=max_init_filters))
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_size',
                                                           lower=min_kernel_size,
                                                           upper=max_kernel_size))
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_stride',
                                                           lower=min_stride,
                                                           upper=max_stride))
        cs.add_hyperparameter(UniformIntegerHyperparameter('conv_kernel_padding',
                                                           lower=min_padding,
                                                           upper=max_padding))
        cs.add_hyperparameter(UniformIntegerHyperparameter('pool_size',
                                                           lower=min_pool_size,
                                                           upper=max_pool_size))
        return cs


class _DenseLayer(nn.Sequential):
    def __init__(self,
                 num_input_features: int,
                 activation: str,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 bn_args: Dict[str, Any]):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features, **bn_args)),
        self.add_module('relu1', _activations[activation]()),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate, **bn_args)),
        self.add_module('relu2', _activations[activation]()),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 activation: str,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 bn_args: Dict[str, Any]):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features=num_input_features + i * growth_rate,
                                activation=activation,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                bn_args=bn_args)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self,
                 num_input_features: int,
                 activation: str,
                 num_output_features: int,
                 pool_size: int,
                 bn_args: Dict[str, Any]):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, **bn_args))
        self.add_module('relu', _activations[activation]())
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))


class DenseNetImageBackbone(BaseBackbone):
    supported_tasks = {"image_classification", "image_regression"}

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.bn_args = {"eps": 1e-5, "momentum": 0.1}

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        channels, iw, ih = input_shape

        growth_rate = self.config['growth_rate']
        block_config = [self.config['layer_in_block_%d' % (i + 1)] for i in range(self.config['blocks'])]
        num_init_features = 2 * growth_rate
        bn_size = 4
        drop_rate = self.config['dropout'] if self.config['use_dropout'] else 0

        image_size, min_image_size = min(iw, ih), 1

        division_steps = math.floor(math.log2(image_size) - math.log2(min_image_size) - 1e-5) + 1

        if division_steps > len(block_config) + 1:
            # First convolution
            features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features, **self.bn_args)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
            division_steps -= 2
        else:
            features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                activation=self.config["activation"],
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                bn_args=self.bn_args)
            features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    activation=self.config["activation"],
                                    num_output_features=num_features // 2,
                                    pool_size=2 if i > len(block_config) - division_steps else 1,
                                    bn_args=self.bn_args)
                features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        features.add_module('last_norm', nn.BatchNorm2d(num_features, **self.bn_args))
        self.backbone = features
        return features

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'DenseNetImageBackbone',
            'name': 'DenseNetImageBackbone',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_growth_rate: int = 12,
                                        max_growth_rate: int = 40,
                                        min_num_blocks: int = 3,
                                        max_num_blocks: int = 4,
                                        min_num_layers: int = 4,
                                        max_num_layers: int = 64) -> ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        growth_rate_hp = UniformIntegerHyperparameter('growth_rate',
                                                      lower=min_growth_rate,
                                                      upper=max_growth_rate)
        cs.add_hyperparameter(growth_rate_hp)

        blocks_hp = UniformIntegerHyperparameter('blocks',
                                                 lower=min_num_blocks,
                                                 upper=max_num_blocks)
        cs.add_hyperparameter(blocks_hp)

        activation_hp = CategoricalHyperparameter('activation',
                                                  choices=list(_activations.keys()))
        cs.add_hyperparameter(activation_hp)

        use_dropout = CategoricalHyperparameter('use_dropout', choices=[True, False])
        dropout = UniformFloatHyperparameter('dropout',
                                             lower=0.0,
                                             upper=1.0)
        cs.add_hyperparameters([use_dropout, dropout])
        cs.add_condition(CS.EqualsCondition(dropout, use_dropout, True))

        for i in range(1, max_num_blocks + 1):
            layer_hp = UniformIntegerHyperparameter('layer_in_block_%d' % i,
                                                    lower=min_num_layers,
                                                    upper=max_num_layers)
            cs.add_hyperparameter(layer_hp)

            if i > min_num_blocks:
                cs.add_condition(CS.GreaterThanCondition(layer_hp, blocks_hp, i - 1))

        return cs


class _SkipConnection(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int):
        super().__init__()

        self.s1 = nn.Sequential()
        self.s1.add_module("Skip_1_AvgPool",
                           nn.AvgPool2d(1, stride=stride))
        self.s1.add_module("Skip_1_Conv2",
                           nn.Conv2d(in_channels,
                                     int(out_channels / 2),
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False))

        self.s2 = nn.Sequential()
        self.s2.add_module("Skip_2_AvgPool",
                           nn.AvgPool2d(1, stride=stride))
        self.s2.add_module("Skip_2_Conv",
                           nn.Conv2d(in_channels,
                                     int(out_channels / 2) if out_channels % 2 == 0 else int(out_channels / 2) + 1,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False))

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = torch.relu(x)
        out1 = self.s1(out1)

        out2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        out2 = self.s2(out2)

        out = torch.cat([out1, out2], dim=1)
        out = self.batch_norm(out)

        return out


class _ResidualBranch(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 branch_index: int):
        super().__init__()

        self.residual_branch = nn.Sequential()

        self.residual_branch.add_module(f"Branch_{branch_index}:ReLU_1",
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module(f"Branch_{branch_index}:Conv_1",
                                        nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=round(kernel_size / 3),
                                                  bias=False))
        self.residual_branch.add_module(f"Branch_{branch_index}:BN_1",
                                        nn.BatchNorm2d(out_channels))
        self.residual_branch.add_module(f"Branch_{branch_index}:ReLU_2",
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module(f"Branch_{branch_index}:Conv_2",
                                        nn.Conv2d(out_channels,
                                                  out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=1,
                                                  padding=round(kernel_size / 3),
                                                  bias=False))
        self.residual_branch.add_module(f"Branch_{branch_index}:BN_2",
                                        nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_branch(x)


class _BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 res_branches: int,
                 stride: int,
                 shake_config: Tuple[bool, bool, bool],
                 block_config: Dict[str, Any]):
        super().__init__()

        self.block_config = block_config
        self.shake_config = shake_config

        self.branches = nn.ModuleList(
            [_ResidualBranch(in_channels, out_channels, kernel_size, stride, branch + 1) for branch in
             range(res_branches)])

        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip.add_module("Skip_connection",
                                 _SkipConnection(in_channels, out_channels, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.branches) == 1:
            out = self.branches[0](x)
            if self.block_config["apply_shakeDrop"]:
                alpha, beta = generate_alpha_beta_single(out.size(),
                                                         self.shake_config if self.training else (False, False, False),
                                                         x.is_cuda)
                out = shake_drop(out, alpha, beta, self.block_config["death_rate"], self.training)
        else:
            if self.block_config["apply_shakeShake"]:
                alpha, beta = generate_alpha_beta(len(self.branches), x.size(0),
                                                  self.shake_config if self.training else (False, False, False),
                                                  x.is_cuda)
                branches = [self.branches[i](x) for i in range(len(self.branches))]
                out = shake_shake(alpha, beta, *branches)
            else:
                out = sum([self.branches[i](x) for i in range(len(self.branches))])

        return out + self.skip(x)


class _ResidualGroup(nn.Module):
    def __init__(self,
                 block: nn.Module,
                 in_channels: int,
                 out_channels: int,
                 n_blocks: int,
                 kernel_size: int,
                 res_branches: int,
                 stride: int,
                 shake_config: Tuple[bool, bool, bool],
                 block_config: Dict[str, Any]):
        super().__init__()
        self.group = nn.Sequential()
        self.n_blocks = n_blocks

        # The first residual block in each group is responsible for the input downsampling
        self.group.add_module("Block_1",
                              block(in_channels,
                                    out_channels,
                                    kernel_size,
                                    res_branches,
                                    stride=stride,
                                    shake_config=shake_config,
                                    block_config=block_config))

        # The following residual block do not perform any downsampling (stride=1)
        for block_index in range(2, n_blocks + 1):
            block_name = f"'Block_{block_index}"
            self.group.add_module(block_name,
                                  block(out_channels,
                                        out_channels,
                                        kernel_size,
                                        res_branches,
                                        stride=1,
                                        shake_config=shake_config,
                                        block_config=block_config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group(x)


class ResNetImageBackbone(BaseBackbone):
    supported_tasks = {"image_classification", "image_regression"}

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.forward_shake = True
        self.backward_shake = True
        self.shake_image = True

        self.block_config = {
            "apply_shakeDrop": True,
            "apply_shakeShake": True,
            "death_rate": self.config["death_rate"]
        }

        self.num_residual_blocks = dict([
            (f"Group_{i + 1}", self.config[f"num_residual_blocks_{i + 1}"])
            for i in range(self.config["num_main_blocks"])])

        self.widen_factors = dict([
            (f"Group_{i + 1}", self.config[f"widen_factor_{i + 1}"])
            for i in range(self.config["num_main_blocks"])])

        self.res_branches = dict([
            (f"Group_{i + 1}", self.config['res_branches_%i' % (i + 1)])
            for i in range(self.config["num_main_blocks"])])

        self.kernel_sizes = dict([
            (f"Group_{i + 1}", 3)
            for i in range(self.config["num_main_blocks"])])

        self.shake_config = (self.forward_shake, self.backward_shake,
                             self.shake_image)

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        backbone = nn.Sequential()

        block = _BasicBlock
        C, H, W = input_shape
        im_size = max(H, W)

        backbone.add_module("Conv_0",
                            nn.Conv2d(in_channels=C,
                                      out_channels=self.config["initial_filters"],
                                      kernel_size=7 if im_size > 200 else 3,
                                      stride=2 if im_size > 200 else 1,
                                      padding=3 if im_size > 200 else 1,
                                      bias=False))
        backbone.add_module("BN_0",
                            nn.BatchNorm2d(self.config["initial_filters"]))

        if im_size > 200:
            backbone.add_module("ReLU_0", nn.ReLU(inplace=True))
            backbone.add_module("Pool_0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        feature_maps_in = int(round(self.config["initial_filters"] * self.widen_factors["Group_1"]))
        backbone.add_module("Group_1",
                            _ResidualGroup(block=block,
                                           in_channels=self.config["initial_filters"],
                                           out_channels=feature_maps_in,
                                           n_blocks=self.num_residual_blocks["Group_1"],
                                           kernel_size=self.kernel_sizes["Group_1"],
                                           res_branches=self.res_branches["Group_1"],
                                           stride=1,
                                           shake_config=self.shake_config,
                                           block_config=self.block_config))

        for main_block_nr in range(2, self.config["num_main_blocks"] + 1):
            feature_maps_out = int(round(feature_maps_in * self.widen_factors[f"Group_{main_block_nr}"]))
            backbone.add_module(f"Group_{main_block_nr}",
                                _ResidualGroup(block=block,
                                               in_channels=feature_maps_in,
                                               out_channels=feature_maps_out,
                                               n_blocks=self.num_residual_blocks[f"Group_{main_block_nr}"],
                                               kernel_size=self.kernel_sizes[f"Group_{main_block_nr}"],
                                               res_branches=self.res_branches[f"Group_{main_block_nr}"],
                                               stride=2,
                                               shake_config=self.shake_config,
                                               block_config=self.block_config))

            feature_maps_in = feature_maps_out

        backbone.apply(initialize_weights)
        self.backbone = backbone
        return backbone

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'ResNetImageBackbone',
            'name': 'ResNetImageBackbone',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        num_main_blocks_min: int = 1,
                                        num_main_blocks_max: int = 8,
                                        num_residual_blocks_min: int = 1,
                                        num_residual_blocks_max: int = 16,
                                        num_initial_filters_min: int = 8,
                                        num_initial_filters_max: int = 32,
                                        widen_factor_min: int = 0.5,
                                        widen_factor_max: int = 4,
                                        num_res_branches_min: int = 1,
                                        num_res_branches_max: int = 5,
                                        **kwargs: Any) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        num_main_blocks_hp = CS.UniformIntegerHyperparameter("num_main_blocks",
                                                             lower=num_main_blocks_min,
                                                             upper=num_main_blocks_max)
        cs.add_hyperparameter(num_main_blocks_hp)

        initial_filters_hp = CS.UniformIntegerHyperparameter("initial_filters",
                                                             lower=num_initial_filters_min,
                                                             upper=num_initial_filters_max)
        cs.add_hyperparameter(initial_filters_hp)

        death_rate_hp = CS.UniformFloatHyperparameter("death_rate",
                                                      lower=0,
                                                      upper=1,
                                                      log=False)
        cs.add_hyperparameter(death_rate_hp)

        for i in range(1, num_main_blocks_max + 1):
            blocks_hp = CS.UniformIntegerHyperparameter(f"num_residual_blocks_{i}",
                                                        lower=num_residual_blocks_min,
                                                        upper=num_residual_blocks_max)
            cs.add_hyperparameter(blocks_hp)

            widen_hp = CS.UniformFloatHyperparameter(f"widen_factor_{i}",
                                                     lower=widen_factor_min,
                                                     upper=widen_factor_max,
                                                     log=True)
            cs.add_hyperparameter(widen_hp)

            branches_hp = CS.UniformIntegerHyperparameter(f"res_branches_{i}",
                                                          lower=num_res_branches_min,
                                                          upper=num_res_branches_max)
            cs.add_hyperparameter(branches_hp)

            if i > num_main_blocks_min:
                cs.add_condition(CS.GreaterThanCondition(blocks_hp, num_main_blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(widen_hp, num_main_blocks_hp, i - 1))
                cs.add_condition(CS.GreaterThanCondition(branches_hp, num_main_blocks_hp, i - 1))
        return cs
