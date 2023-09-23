import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import math
import lightning as L


class BasicNN(L.LightningModule):
    def __init__(self, layer_sizes, dataset, batch_size=500, learning_rate=0.05):
        super().__init__()

        def create_layers(sizes) -> nn.Sequential:
            layers = []
            for i, (input_size, output_size) in enumerate(sizes):
                linear = nn.Linear(input_size, output_size)
                nn.init.xavier_uniform_(linear.weight)
                layers.append(linear)
                if i != len(sizes) - 1:
                    layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)

        sizes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.net = create_layers(sizes)
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train_dataloader(self):
        return DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size)

    def forward(self, X: torch.Tensor):
        return self.net(X)

    def predict(self, X: torch.Tensor):
        return self(X)

    def training_step(self, batch, batch_idx):
        X, label = batch
        output = self.forward(X)
        print(torch.sum((label - output) ** 2))
        return F.mse_loss(output, label)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
        # return SGD(self.parameters(), lr=self.learning_rate, momentum=0.3)
