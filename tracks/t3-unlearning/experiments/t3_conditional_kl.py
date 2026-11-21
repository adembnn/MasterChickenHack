"""Ablate the weight of conditional non-target KL with fixed teacher targets."""

import torch
from torch.nn import functional as F


def conditional_kl(student_logits, target_logits, pseudo_labels, extra_weight=0.0, temperature=2.0):
    """Mean KL(target || student) plus extra_weight times KL on the other classes.

    Both inputs are raw logits at T=1. Like the contest metric, apply temperature
    before computing KL. The caller retains its existing outer loss weights.
    """
    if temperature <= 0 or extra_weight < 0:
        raise ValueError("Expected positive temperature and nonnegative extra weight")
    if student_logits.ndim != 2 or student_logits.shape != target_logits.shape:
        raise ValueError("Expected two matching batches of class logits")
    batch, classes = student_logits.shape
    if pseudo_labels.shape != (batch,) or classes < 2 or batch == 0:
        raise ValueError("Expected one pseudo-label per example and at least two classes")

    student = student_logits.float() / temperature
    teacher = target_logits.detach().float() / temperature
    other = torch.ones_like(student, dtype=torch.bool)
    other.scatter_(1, pseudo_labels[:, None], False)
    target_other = teacher[other].reshape(batch, classes - 1).log_softmax(dim=1)
    student_other = student[other].reshape(batch, classes - 1).log_softmax(dim=1)
    conditional = F.kl_div(student_other, target_other, reduction="batchmean", log_target=True)
    full = F.kl_div(student.log_softmax(1), teacher.log_softmax(1), reduction="batchmean", log_target=True)
    return full + extra_weight * conditional
