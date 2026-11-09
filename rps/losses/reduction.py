import torch

def weighted_mean(loss_vec, weight_vec, eps=1e-06):
    w_sum = weight_vec.sum()
    if w_sum.item() < eps:
        return loss_vec.new_tensor(0.0)
    return (loss_vec * weight_vec).sum() / (w_sum + eps)
