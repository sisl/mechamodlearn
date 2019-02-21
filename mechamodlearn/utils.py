# File: utils.py
#

import random
import sys
from contextlib import contextmanager
import resource
import numpy as np
from numba import jit
import timeit
import torch


def bfill_lowertriangle(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A


def peak_memory_mb() -> float:
    """
    Get peak memory usage for this process, as measured by
    max-resident-set size:
    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size
    Only works on OSX and Linux, returns 0.0 otherwise.
    """
    if resource is None or sys.platform not in ('linux', 'darwin'):
        return 0.0

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore

    if sys.platform == 'darwin':
        # On OSX the result is in bytes.
        return peak / 1_000_000

    else:
        # On Linux the result is in kilobytes.
        return peak / 1_000


def set_rng_seed(rng_seed: int) -> None:
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)


def wrap_to_pi(inp, mask=None):
    """Wraps to [-pi, pi)"""
    if mask is None:
        mask = inp.new_ones(1, inp.size(1))

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    mask = mask.to(dtype=inp.dtype, device=inp.device)
    val = torch.fmod((inp + np.pi) * mask, 2 * np.pi)
    neg_mask = (val * mask) < 0
    val = val + 2 * np.pi * neg_mask.to(val.dtype)
    val = (val - np.pi)
    inp = (1 - mask) * inp + mask * val
    return inp


def diffangles(inp1, inp2, mask=None):
    """
    computes the difference between two
    angles [in rad] accounting for the
    branch cut at pi
    """

    return wrap_to_pi(inp1 - inp2, mask=mask)


def require_and_zero_grads(vs):
    for v in vs:
        v.requires_grad_(True)
        try:
            v.grad.zero_()
        except AttributeError:
            pass


@contextmanager
def temp_require_grad(vs):
    prev_grad_status = [v.requires_grad for v in vs]
    require_and_zero_grads(vs)
    yield
    for v, status in zip(vs, prev_grad_status):
        v.requires_grad_(status)


@jit(nopython=True, parallel=True)
def time_series_norm(xs, dt, p=2):
    """
    Computes the Lp norm for a time series

    See Lee & Verleysen, Generalization of the Lp norm for time series and its
    application to Self-Organizing Maps
    https://pdfs.semanticscholar.org/713c/2c5546e34ae25d808d375fc071551681c7ec.pdf
    """
    assert xs.ndim == 1
    sumd = 0.0
    for i in range(xs.shape[0]):
        if i == 0:
            prevx = 0.0
        else:
            prevx = xs[i - 1]

        if i == xs.shape[0] - 1:
            nextx = 0.0
        else:
            nextx = xs[i + 1]
        if xs[i] * prevx <= 0:
            Am1 = dt / 2 * np.abs(xs[i])
        else:
            Am1 = dt / 2 * xs[i]**2 / (np.abs(xs[i]) + np.abs(prevx))

        if xs[i] * nextx <= 0.0:
            Ap1 = dt / 2 * np.abs(xs[i])
        else:
            Ap1 = dt / 2 * xs[i]**2 / (np.abs(xs[i]) + np.abs(nextx))

        sumd += (Am1 + Ap1)**p

    return sumd**(1 / p)


class Timer(object):

    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start
