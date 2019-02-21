# File: dataset.py
#
import torch

from torch.utils.data import dataset

from mechamodlearn.odesolver import odeint, ActuatedODEWrapper
from mechamodlearn import utils


class ActuatedTrajectoryDataset(dataset.TensorDataset):

    def __init__(self, traj_q_T_B, traj_v_T_B, traj_u_T_B):
        """
        Arguments:
        - `traj_q_T_B`: B trajectories generalized positions of timesteps T
        - `traj_v_T_B`: B trajectories generalized velocities of timesteps T
        - `traj_u_T_B`: B trajectories generalized torques of timesteps T
        """
        self.q_B_T = traj_q_T_B.transpose(1, 0)
        self.v_B_T = traj_v_T_B.transpose(1, 0)
        self.u_B_T = traj_u_T_B.transpose(1, 0)

        assert self.q_B_T.size(0) == self.v_B_T.size(0) == self.u_B_T.size(0)

    def __len__(self):
        return self.q_B_T.size(0)

    def __getitem__(self, index):
        return (self.q_B_T[index], self.v_B_T[index], self.u_B_T[index])

    @classmethod
    def FromSystem(cls, system, q_B, v_B, u_T_B, t_points, method='rk4'):
        """
        Get trajectories given initial conditions, list of torques to apply
        for a given system and time
        """
        assert q_B.size(0) == v_B.size(0) == u_T_B.size(1)
        assert len(t_points) == u_T_B.size(0)

        if not isinstance(system, ActuatedODEWrapper):
            system = ActuatedODEWrapper(system)

        with torch.no_grad():
            q_T_B, v_T_B = odeint(system, (q_B, v_B), t_points, u=u_T_B, method=method,
                                  transforms=(lambda x: utils.wrap_to_pi(x, system.thetamask),
                                              lambda x: x))

        # wrap angles
        q_T_B = utils.wrap_to_pi(q_T_B.view(-1, system._qdim), system.thetamask).view(
            len(t_points), -1, system._qdim)
        return cls(q_T_B, v_T_B, u_T_B)


class ODEPredDataset(dataset.Dataset):

    def __init__(self, qs: list, vs: list, ulist: list):
        """
        Arguments:
        - `qs`: list of len T, containing batches of B
        - `vs`: ditto
        - `ulist`: ditto
        """
        assert all(qs[0].size(0) == q.size(0) for q in qs)
        assert all(vs[0].size(0) == qd.size(0) for qd in vs)
        assert all(ulist[0].size(0) == u.size(0) for u in ulist)
        assert qs[0].size(0) == vs[0].size(0) == ulist[0].size(0)  # same batch size
        self.qs_tensors = qs
        self.vs_tensors = vs
        self.u_tensors = ulist

    def __getitem__(self, index):
        return (tuple(q[index]
                      for q in self.qs_tensors), tuple(v[index] for v in self.vs_tensors), tuple(
                          u[index] for u in self.u_tensors))

    def __len__(self):
        return self.qs_tensors[0].size(0)
