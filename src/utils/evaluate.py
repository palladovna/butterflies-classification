import torch
from src.utils.metrics import accuracy, precision, recall, f1


def evaluate(model, data_loader, device):
    """
    Evaluate model on test_loader.
    Returns dict of metrics.
    """
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().tolist())
            labels.extend(targets.cpu().tolist())

    return {
        "accuracy": accuracy(preds, labels),
        "precision": precision(preds, labels),
        "recall": recall(preds, labels),
        "f1": f1(preds, labels)
    }
