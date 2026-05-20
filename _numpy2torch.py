import torch


def zeros(shape):
    return torch.zeros(shape, requires_grad=False, dtype=torch.double)


def tile(A, reps):
    return torch.tile(A, dims=(reps,))


def swapaxes(a, axis1, axis2):
    return torch.swapaxes(a, axis0=axis1, axis1=axis2)


def mean(a, axis):
    return torch.mean(a, dim=axis)


def copy(A):
    return torch.clone(A)


def roll(A, shift, axis):
    return torch.roll(A, shifts=shift, dims=axis)


def mod(x1, x2):
    return torch.remainder(x1, other=x2)


def flip(A, axis):
    return torch.flip(A, dims=(axis,))


def cumsum(A, axis):
    return torch.cumsum(A, dim=axis)


def hsplit(A, indices_or_sections):
    return torch.hsplit(A, indices_or_sections)


def sum(A, axis):
    return torch.sum(A, dim=axis)


def hstack(tup):
    return torch.hstack(tensors=tup)


def zeros_like(a):
    return torch.zeros_like(input=a)
