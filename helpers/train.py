import torch
from sklearn.metrics import f1_score
import copy


def train_one_epoch(train_data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = model.loss(out[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()



@torch.no_grad()
def evaluate(data, model):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    true = data.y

    results = {}
    for split_name, mask in {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}.items():
        y_true = true[mask]
        y_pred = pred[mask]
        logits = out[mask]
        loss = model.loss(logits, y_true).item()
        f1 = f1_score(y_true.cpu(), y_pred.cpu(), average="weighted", zero_division=0)
        results[split_name] = {"loss": loss, "f1": f1}

    return results


def train(model, train_data, optimizer, scheduler, params, verbose):
    device = torch.device(params.device)
    model.to(device)
    train_data = train_data.to(device)

    best_val_loss = float("inf")
    best_state_dict = None
    best_metrics = None
    epochs_no_improve = 0

    for epoch in range(1, params.n_epochs + 1):
                    
        train_one_epoch(train_data, model, optimizer)
        full_metrics = evaluate(train_data, model)

        val_loss = full_metrics["val"]["loss"]
        scheduler.step(val_loss)
        
        

        if epoch % params.print_every == 0 and verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss = {full_metrics['train']['loss']:.4f} | Train F1 = {full_metrics['train']['f1']:.4f} | "
                f"Val Loss = {full_metrics['val']['loss']:.4f} | Val F1 = {full_metrics['val']['f1']:.4f}"
            )

        if params.epoch_patience is not None:
            if val_loss < best_val_loss - params.min_delta:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                best_metrics = full_metrics
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= params.epoch_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch:03d}")
                    break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        full_metrics = best_metrics

    return full_metrics
