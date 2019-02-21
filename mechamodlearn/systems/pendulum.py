# File: pendulum.py
#

import torch

from mechamodlearn.rigidbody import AbstractRigidBody
from mechamodlearn.models import ControlAffineLinearForce, ViscousJointDampingForce, GeneralizedForces


class SimplePendulumMM(torch.nn.Module):

    def __init__(self, l):
        """
        Arguments:
        - `l`: length of the pendulum
        """
        super().__init__()

        assert l.shape == (1,), l.shape
        if not isinstance(l, torch.nn.Parameter):
            l = torch.nn.Parameter(l)

        self._l = l

    def forward(self, q):
        return self._l * torch.ones_like(q).unsqueeze(2) + 0. * q.unsqueeze(2)


class SimplePendulumV(torch.nn.Module):

    def __init__(self, gl):
        """
        Arguments:
        - `gl`: gravity \times length
        """
        super().__init__()

        assert gl.shape == (1,)
        if not isinstance(gl, torch.nn.Parameter):
            gl = torch.nn.Parameter(gl)

        self._gl = gl

    def forward(self, q):
        return -self._gl * torch.cos(q)


class SimplePendulum(AbstractRigidBody, torch.nn.Module):

    def __init__(self, params=None):

        self._qdim = 1
        self._udim = 1

        self._thetamask = torch.tensor([1])

        if params is None:
            params = torch.abs(torch.randn(2))

        assert params.shape == (2,)

        super().__init__()

        l, gl = torch.unbind(params)

        self._mass_matrix = SimplePendulumMM(l.unsqueeze(0))
        self._potential = SimplePendulumV(gl.unsqueeze(0))

    def mass_matrix(self, q):
        return self._mass_matrix(q)

    def potential(self, q):
        return self._potential(q)

    @property
    def thetamask(self):
        return self._thetamask

    def forward(self, q, v, u=None):
        return self.solve_euler_lagrange(q, v, u)


class ActuatedSimplePendulum(SimplePendulum):

    def __init__(self, params=None):
        if params is None:
            params = torch.abs(torch.randn(3))

        super().__init__(params=params[:2])
        self._lin_force = ControlAffineLinearForce(
            torch.diag_embed(params[-1].reshape(self._qdim, self._qdim)))

    def generalized_force(self, q, v, u):
        return self._lin_force(q, v, u)


class ActuatedDampedPendulum(ActuatedSimplePendulum):

    def __init__(self, params=None):
        if params is None:
            params = torch.abs(torch.randn(4))

        super().__init__(params=params[:3])
        self._visc_force = ViscousJointDampingForce(params[-1].reshape(1, self._qdim))
        self._generalized_force = GeneralizedForces([self._lin_force, self._visc_force])

    def generalized_force(self, q, v, u):
        return self._generalized_force(q, v, u)
