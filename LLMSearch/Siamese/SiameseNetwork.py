import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

class BaseNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = (self.layer2(x))
        #x = (self.layer3(x))
        return x

class BaseEqualNetwork(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        """
        start from equal output
        """
        self.layer1 = nn.Linear(input_dim, input_dim)
        self.layer1.weight = nn.Parameter(torch.eye(input_dim))
        self.layer1.bias = nn.Parameter(torch.zeros(input_dim))

        self.layer2 = nn.Linear(input_dim, input_dim)
        self.layer2.weight = nn.Parameter(torch.eye(input_dim))
        self.layer2.bias = nn.Parameter(torch.zeros(input_dim))

        self.layer3 = nn.Linear(input_dim, input_dim)
        self.layer3.weight = nn.Parameter(torch.eye(input_dim))
        self.layer3.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = (self.layer2(x))
        #x = (self.layer3(x))
        return x

class SiameseNetwork(pl.LightningModule):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        #self.base_network = BaseNetwork(input_dim, hidden_dim)
        self.base_network = BaseEqualNetwork(input_dim)
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        input1, input2, labels = batch
        output1, output2 = self.forward(input1, input2)
        loss = self.loss(output1, output2, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input1, input2, labels = batch
        output1, output2 = self.forward(input1, input2)
        loss = self.loss(output1, output2, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
