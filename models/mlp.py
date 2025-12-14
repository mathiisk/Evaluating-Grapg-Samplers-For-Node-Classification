import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple multilayer perceptron for node or feature classification

    inputs:
        in_dim (int): input feature dimension
        out_dim (int): number of output classes
        params: config object with model settings

    returns:
        logits for each class
    """

    def __init__(self, in_dim, out_dim, params):
        super().__init__()

        self.dropout = params.dropout
        self.n_classes = out_dim
        self.device = params.device

        # optional batch norm on output
        if params.use_batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.bn = None

        # input embedding layer
        self.embedding = nn.Linear(in_dim, params.hidden_dim)

        # hidden fully connected layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(params.num_layers):
            self.hidden_layers.append(
                nn.Linear(params.hidden_dim, params.hidden_dim)
            )

        # final output layer
        self.out_layer = nn.Linear(params.hidden_dim, out_dim)

        # initialize all weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset all learnable parameters and uses xavier init for weights and zeros for bias
        """
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)

        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, feature, edge_index=None):
        """
        Forward pass of the mlp. Edge_index is ignored here.
        inputs:
            feature (tensor): input features [num_nodes, in_dim]
            edge_index: unused, kept for compatibility

        returns:
            tensor: output logits
        """
        # input embedding with activation and dropout
        h = self.embedding(feature)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # process through hidden layers
        for layer in self.hidden_layers:
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # output layer (no relu, no dropout)
        h = self.out_layer(h)

        # apply batch norm if enabled
        if self.bn:
            h = self.bn(h)

        return h

    def loss(self, pred, label):
        """
        Compute weighted cross entropy loss: helps when classes are imbalanced.

        inputs:
            pred (tensor): predicted logits
            label (tensor): ground truth labels

        returns:
            tensor: scalar loss value
        """
        # calculating label weights for weighted loss computation
        V = label.size(0)

        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero(as_tuple=False)].squeeze()

        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count

        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
