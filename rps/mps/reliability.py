"""Token reliability on the existing weak/strong positional correspondence."""
import math

import torch
import torch.nn.functional as F


def confidence_from_logits(logits, temperature=2.):
    probability = F.softmax(logits.float()/float(temperature), dim=-1)
    entropy = -(probability*probability.clamp_min(1e-12).log()).sum(-1)
    return 1.-entropy/(math.log(probability.size(-1))+1e-12)


def stability_from_logits(weak_ids, strong_ids, strong_logits, valid):
    """Inputs use the training pipeline's EOS/PAD and insertion mapping."""
    length = min(weak_ids.size(1), strong_logits.size(1))
    score = torch.zeros_like(weak_ids, dtype=torch.float32)
    target_probability = F.softmax(strong_logits[:,:length].float(), -1).gather(
        -1, weak_ids[:,:length,None]).squeeze(-1)
    same = (strong_ids[:,:length]==weak_ids[:,:length]).float()
    score[:,:length] = torch.maximum(target_probability,same)
    return torch.where(valid,score,torch.ones_like(score))


def agreement_from_logits(student_logits, weak_ids):
    return F.softmax(student_logits.float(), -1).gather(-1,weak_ids.unsqueeze(-1)).squeeze(-1)
