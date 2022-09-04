# this code is not my own it was adapted from Ayush Thukar

import os

import pytorch_lightning as pl
import torch

import torchvision.models as models
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional
from torchmetrics import Accuracy


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, transfer=True):
        super().__init__()

        # log hyperparameters in wandb
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes

        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=transfer)

        if transfer:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        n_sizes = self._get_conv_output(input_shape)
        self.classifier = nn.Linear(n_sizes, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        """
        Feature extraction is a torchvision.models method that
        allows us to access intermediate transformations of the
        inputs.


        This is a PTL feature really and the code it helps out is
        best_model_wts = copy.deepcopy(model.state_dict())

        """
        x = self.feature_extractor(x)
        return x

    def _get_conv_output(self, shape):
        """
        returns the size of the output tensor going into the Linear layer from the conv block
        """
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

  
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # Linear layer we made earlier

        return x

    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        acc = self.accuracy(out, gt)

        # record in wandb
        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        self.log("val/loss", loss)

        acc = self.accuracy(out, gt)
        self.log("val/acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)
        return {"loss": loss, "outputs": out, "gt": gt}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        output = torch.cat([x["outputs"] for x in outputs], dim=0)
        gts = torch.cat([x["gt"] for x in outputs], dim=0)

        # ~self.log("test/loss_epoch", loss, on_step = False, on_epoch = True) in test_step above
        self.log("test/loss", loss)
        acc = self.accuracy(output, gts)
        self.log("test/acc", acc)

        self.test_gts = gts
        self.test_output = output

        model_filename = "food_resnet.onnx"
        torch.onnx.export(self, model_filename = model_filename)
        wandb.save(model_filename)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
