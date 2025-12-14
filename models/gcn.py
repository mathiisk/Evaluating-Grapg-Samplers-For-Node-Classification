import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    """
    Single gcn layer used inside the main model

    inputs:
        in_dim (int): size of input node features
        out_dim (int): size of output node features
        activation: activation function (currently relu is used)
        dropout (float): dropout probability
        batch_norm (bool): whether to use batch normalization
        device: torch device
        residual (bool): add residual connection if dims match

    returns:
        tensor with updated node features
    """
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, device, residual=False):
        super().__init__()

        # only use residual if dims match, otherwise it would break
        self.residual = residual and (in_dim == out_dim)

        # saving stuff we might need later
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.n_classes = out_dim
        self.device = device

        # main graph convolution
        self.conv = GCNConv(in_dim, out_dim)

        # batch norm is optional
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.bn = None

    def forward(self, x, edge_index):
        """
        forward pass of one gcn layer
        """

        # keep input for residual connection
        x_in = x

        # graph convolution step
        x = self.conv(x, edge_index)

        # apply batch norm if enabled
        if self.bn:
            x = self.bn(x)

        # non linearity
        x = F.relu(x)

        # add residual if we can
        if self.residual:
            x = x + x_in

        # final dropout
        return self.dropout(x)


class OutputMLP(nn.Module):
    """
    simple mlp used as output classifier

    inputs:
        input_dim (int): size of input features
        output_dim (int): number of classes
        num_layers (int): number of hidden layers

    returns:
        logits for each class
    """
    def __init__(self, input_dim, output_dim, num_layers=2):
        super().__init__()

        layers = []
        current_dim = input_dim

        # build hidden layers
        for _ in range(num_layers):
            next_dim = current_dim // 2   # reduce size slowly
            layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim

        # last layer maps to number of classes
        layers.append(nn.Linear(current_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers

    def forward(self, x):
        """
        Forward pass of the mlp
        """
        # apply relu to all but last layer
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # final linear layer (no relu here)
        x = self.layers[-1](x)
        return x


class GCN(nn.Module):
    """
    Full gcn model with embedding layer, gcn stack,
    and an mlp classifier at the end

    inputs:
        in_dim (int): input feature size
        out_dim (int): number of classes
        params: config object with model hyperparams

    returns:
        class logits for each node
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        params,
    ):
        super().__init__()

        self.n_classes = out_dim
        self.device = params.device

        # first linear layer to get node embeddings
        self.embed = nn.Linear(in_dim, params.hidden_dim)

        # stack of gcn layers
        self.layers = nn.ModuleList([
            GCNLayer(
                params.hidden_dim,
                params.hidden_dim,
                params.activation,
                params.dropout,
                params.use_batchnorm,
                residual=params.use_residual,
                device=params.device
            )
            for _ in range(params.num_layers)
        ])

        # final classifier head
        self.classifier = OutputMLP(
            params.hidden_dim,
            out_dim,
            num_layers=params.num_layers
        )

        # init weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all learnable weights
        mainly used when restarting training
        """
        # init embedding layer
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

        # reset gcn layers
        for layer in self.layers:
            layer.conv.reset_parameters()
            if layer.bn:
                layer.bn.reset_parameters()

        # init mlp layers
        for layer in self.classifier.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index):
        """
        forward pass of the full gcn model

        inputs:
            x (tensor): node features [num_nodes, in_dim]
            edge_index (tensor): graph connectivity

        returns:
            tensor: class logits for each node
        """
        # project input features
        x = self.embed(x)

        # dropout on embeddings
        x = F.dropout(x, p=0.5, training=self.training)

        # pass through gcn stack
        for layer in self.layers:
            x = layer(x, edge_index)

        # final prediction
        return self.classifier(x)

    def loss(self, pred, label):
        """
        Compute weighted cross entropy loss (due to possible class imbalance)

        inputs:
            pred (tensor): predicted logits
            label (tensor): ground truth labels

        returns:
            tensor: scalar loss value
        """
        # compute class weights to handle imbalance
        V = label.size(0)

        # count how many times each label appears
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero(as_tuple=False)].squeeze()

        # store cluster sizes for all classes
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count

        # higher weight for smaller classes
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross entropy loss
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

    def encode(self, x, edge_index):
        """
        Encode nodes into latent representations: runs the gcn without dropout or classifier,
        and temporarily switches to eval mode. Used when visualizing node embedings.

        inputs:
            x (tensor): node features [num_nodes, in_dim]
            edge_index (tensor): graph connectivity

        returns:
            tensor: node embeddings after gcn layers
        """
        was_training = self.training
        self.eval()

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, edge_index)

        if was_training:
            self.train()

        return x

