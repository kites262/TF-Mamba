import torch


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict:
    """
    logits: (N, C)
    targets: (N,)
    """
    preds = logits.argmax(dim=1)

    correct = (preds == targets).sum().item()
    total = targets.numel()
    oa = correct / total

    precision = []
    recall = []
    f1 = []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return {
        "oa": oa,
        "precision": sum(precision) / num_classes,
        "recall": sum(recall) / num_classes,
        "f1_score": sum(f1) / num_classes,
    }
