import torch
from torch_geometric.data import Data


def subsample_train(data, sampler, train_ratio):
    dev = data.x.device

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    val_idx   = data.val_mask.nonzero(as_tuple=False).view(-1)
    test_idx  = data.test_mask.nonzero(as_tuple=False).view(-1)

    k = max(1, min(train_idx.numel(), int(round(train_ratio * train_idx.numel()))))

    # sample *within train* using your sampler (needs graph structure)
    train_sub = data.subgraph(train_idx)
    sampled_local = sampler.sample(train_sub, k)
    sampled_train = train_idx[sampled_local]

    kept = torch.cat([sampled_train, val_idx, test_idx], dim=0)

    out = data.subgraph(kept)
    out.n_id = kept  # mapping back to original node ids (optional)

    n_tr = sampled_train.numel()
    n_va = val_idx.numel()
    n_te = test_idx.numel()

    out.train_mask = torch.zeros(out.num_nodes, dtype=torch.bool, device=dev)
    out.val_mask   = torch.zeros(out.num_nodes, dtype=torch.bool, device=dev)
    out.test_mask  = torch.zeros(out.num_nodes, dtype=torch.bool, device=dev)

    out.train_mask[:n_tr] = True
    out.val_mask[n_tr:n_tr + n_va] = True
    out.test_mask[n_tr + n_va:n_tr + n_va + n_te] = True

    return out




def create_data_masks(data, params):
    num_nodes = data.num_nodes
    all_indices = torch.randperm(num_nodes, device=data.x.device)

    assert params.train_split + params.val_split + params.test_split <= 1.0

    n_train = int(params.train_split * num_nodes)
    n_val = int(params.val_split * num_nodes)
    n_test = int(params.test_split * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)

    train_mask[all_indices[:n_train]] = True
    val_mask[all_indices[n_train:n_train + n_val]] = True
    test_mask[all_indices[n_train + n_val:n_train + n_val + n_test]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
