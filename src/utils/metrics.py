from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch


def accuracy(preds, labels):
    """
    Simple accuracy for classification.
    """
    return accuracy_score(labels, preds)


def precision(preds, labels, average="macro"):
    """
    Precision for multiclass classification.
    """
    return precision_score(labels, preds, average=average, zero_division=0)


def recall(preds, labels, average="macro"):
    """
    Recall for multiclass classification.
    """
    return recall_score(labels, preds, average=average, zero_division=0)


def f1(preds, labels, average="macro"):
    """
    F1 score for multiclass classification.
    """
    return f1_score(labels, preds, average=average, zero_division=0)


def top_k_accuracy(outputs, targets, k=5):
    """
    Computes Top-K accuracy.
    outputs: logits (tensor, shape [batch, num_classes])
    targets: labels (tensor, shape [batch])
    """
    with torch.no_grad():
        _, pred = outputs.topk(k, dim=1)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred)).sum().item()
        return correct / targets.size(0)
