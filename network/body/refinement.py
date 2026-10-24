"""Predicted segmentation feedback for shared-weight recycling."""

import torch
from torch import nn


class SegmentationPairHypothesis(nn.Module):
    def __init__(self, num_classes: int, pair_dim: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Refinement requires at least one non-null segmentation class")
        # Class zero is the existing segmentation no-object label.
        self.num_classes = num_classes
        self.projection = nn.Linear(num_classes - 1, pair_dim)

    def forward(self, prediction, pair_mask):
        shape = (*pair_mask.shape, self.num_classes - 1)
        if prediction is None:
            hypothesis = self.projection.weight.new_zeros(shape)
        else:
            masks = prediction["pred_masks"].detach().sigmoid()
            classes = prediction["pred_logits"].detach().softmax(-1)[..., 1:]
            if classes.shape[-1] != self.num_classes - 1:
                raise ValueError("Segmentation feedback class count does not match the model")
            hypothesis = torch.einsum("bqi,bqj,bqc->bijc", masks, masks, classes)
        hypothesis = hypothesis.masked_fill(~pair_mask[..., None], 0)
        update = self.projection(hypothesis)
        # Keep the projection in the graph at depth zero, with exactly zero feedback.
        update = update * float(prediction is not None)
        return update.masked_fill(~pair_mask[..., None], 0)


def sample_recycles(max_recycles: int, device):
    """Use the same sampled depth on every distributed training worker."""
    depth = torch.randint(max_recycles + 1, (), device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(depth, src=0)
    return int(depth.item())
