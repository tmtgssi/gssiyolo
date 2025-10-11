import torch
import torch.nn as nn

def cel_loss(vec1, vec2, label=1):
    """
    Compute cosine embedding loss between two batches of vectors.

    Args:
        vec1: Tensor of shape (B, D)
        vec2: Tensor of shape (B, D)
        label: int or Tensor of shape (B,) — 1 for similar, -1 for dissimilar

    Returns:
        Scalar loss
    """
    loss_fn = nn.CosineEmbeddingLoss()
    # Convert int label to tensor if needed
    if isinstance(label, int):
        target = torch.full((vec1.size(0),), float(label), device=vec1.device)
    else:
        target = label.to(vec1.device).float()

    return loss_fn(vec1, vec2, target)
