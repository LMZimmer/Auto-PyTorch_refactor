import unittest

import torch
from torch import nn

from autoPyTorch.pipeline.components.training.losses import get_loss_instance


class LossTest(unittest.TestCase):
    def test_get_no_name(self):
        dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multiclass'}
        loss = get_loss_instance(dataset_properties)
        self.assertTrue(isinstance(loss, nn.Module))

    def test_get_name(self):
        dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multiclass'}
        name = 'CrossEntropyLoss'
        loss = get_loss_instance(dataset_properties, name)
        self.assertIsInstance(loss, nn.Module)
        self.assertEqual(str(loss), 'CrossEntropyLoss()')

    def test_get_name_error(self):
        dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multiclass'}
        name = 'BCELoss'
        try:
            get_loss_instance(dataset_properties, name)
        except ValueError as msg:
            self.assertRegex(str(msg), r"Invalid name entered for task [a-z]+_[a-z]+, "
                                       r"and output type [a-z]+ currently supported losses for task include .*")

    def test_losses(self):
        list_properties = [{'task_type': 'tabular_classification', 'output_type': 'multiclass'},
                           {'task_type': 'tabular_classification', 'output_type': 'binary'},
                           {'task_type': 'tabular_regression', 'output_type': 'continuous'}]
        pred_cross_entropy = torch.randn(4, 4, requires_grad=True)
        list_predictions = [pred_cross_entropy, torch.empty(4).random_(2), torch.randn(4)]
        list_names = [None, 'BCEWithLogitsLoss', None]
        list_targets = [torch.empty(4, dtype=torch.long).random_(4), torch.empty(4).random_(2), torch.randn(4)]
        for dataset_properties, pred, target, name in zip(list_properties, list_predictions, list_targets, list_names):
            loss = get_loss_instance(dataset_properties=dataset_properties, name=name)
            score = loss(pred, target)
            self.assertIsInstance(score, torch.Tensor)
