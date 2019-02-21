# File: transform.py
#
import functools

from more_itertools import windowed

import torch

from mechamodlearn.dataset import ActuatedTrajectoryDataset, ODEPredDataset


def fill_windowed(ls, traj_T_D, chunk_size, step):
    qs_ms = windowed(traj_T_D, chunk_size, step=step)
    for q in qs_ms:
        for i, cq in enumerate(q):
            ls[i].append(cq)


@functools.lru_cache(maxsize=None)
def odepred_transform(traj_dataset: ActuatedTrajectoryDataset, chunk_size: int) -> ODEPredDataset:
    qs_C = [[] for _ in range(chunk_size)]
    vs_C = [[] for _ in range(chunk_size)]
    us_C = [[] for _ in range(chunk_size)]

    if traj_dataset.q_B_T.size(1) > chunk_size:
        for trajq_T_D in traj_dataset.q_B_T.unbind(0):
            fill_windowed(qs_C, trajq_T_D, chunk_size, step=1)
        for trajv_T_D in traj_dataset.v_B_T.unbind(0):
            fill_windowed(vs_C, trajv_T_D, chunk_size, step=1)
        for traju_T_D in traj_dataset.u_B_T.unbind(0):
            fill_windowed(us_C, traju_T_D, chunk_size, step=1)

        for i in range(len(qs_C)):
            qs_C[i] = torch.stack(qs_C[i])  # shape = (B, D)
            vs_C[i] = torch.stack(vs_C[i])  # shape = (B, D)
            us_C[i] = torch.stack(us_C[i])  # shape = (B, D)

    elif traj_dataset.q_B_T.size(1) == chunk_size:
        qs_C = traj_dataset.q_B_T.unbind(1)
        vs_C = traj_dataset.v_B_T.unbind(1)
        us_C = traj_dataset.u_B_T.unbind(1)

    else:
        raise ValueError("{} < chunk size {}".format(traj_dataset.q_B_T.size(1), chunk_size))

    assert qs_C[0].size(0) == vs_C[0].size(0) == us_C[0].size(0)

    assert len(qs_C) == len(vs_C) == len(us_C)

    return ODEPredDataset(qs_C, vs_C, us_C)
