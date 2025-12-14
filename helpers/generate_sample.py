import torch


def subsample_train(data, sampler, train_ratio):
    """
    Subsample the training nodes using a provided sampler.

    Inputs:
        data: PyG data object with train/val/test masks
        sampler: a sampler object with a .sample(data, k) method
        train_ratio: float, fraction of train nodes to keep

    Returns:
        PyG data object with subsampled train nodes and updated masks
    """
    dev = data.x.device

    # get indices of nodes in each split
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    val_idx   = data.val_mask.nonzero(as_tuple=False).view(-1)
    test_idx  = data.test_mask.nonzero(as_tuple=False).view(-1)

    # compute number of train nodes to sample
    k = max(1, min(train_idx.numel(), int(round(train_ratio * train_idx.numel()))))

    # create subgraph with only train nodes
    train_sub = data.subgraph(train_idx)

    # sample train nodes using the provided sampler
    sampled_local = sampler.sample(train_sub, k)
    sampled_train = train_idx[sampled_local]

    # keep sampled train + all val/test nodes
    kept = torch.cat([sampled_train, val_idx, test_idx], dim=0)
    out = data.subgraph(kept)
    out.n_id = kept  # store original node ids

    # update masks for the new subgraph
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
    """
    Randomly split nodes into train/val/test masks based on ratios in params.

    Inputs:
        data: PyG data object
        params: config object with train_split, val_split, test_split

    Returns:
        data: same PyG object with train_mask, val_mask, test_mask updated
    """
    num_nodes = data.num_nodes
    all_indices = torch.randperm(num_nodes, device=data.x.device)

    # sanity check: total fraction should not exceed 1
    assert params.train_split + params.val_split + params.test_split <= 1.0

    n_train = int(params.train_split * num_nodes)
    n_val   = int(params.val_split * num_nodes)
    n_test  = int(params.test_split * num_nodes)

    # initialize boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)

    # assign True to the selected indices
    train_mask[all_indices[:n_train]] = True
    val_mask[all_indices[n_train:n_train + n_val]] = True
    test_mask[all_indices[n_train + n_val:n_train + n_val + n_test]] = True

    # attach masks to data
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    return data
