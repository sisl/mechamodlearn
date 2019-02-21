#
# File: mlacrobot.py
#
import torch

from mechamodlearn.rigidbody import AbstractRigidBody
from mechamodlearn.models import ControlAffineLinearForce, ViscousJointDampingForce, GeneralizedForces


class MultiLinkAcrobotMM(torch.nn.Module):

    def __init__(self, qdim, params=None):
        super().__init__()
        self._qdim = qdim
        if params is None:
            params = torch.abs(torch.randn(self._qdim * 2 + 1))

        assert params.shape == (self._qdim * 2 + 1,)

        if not isinstance(params, torch.nn.Parameter):
            params = torch.nn.Parameter(params)

        self._params = params

    def forward(self, q):
        B = q.size(0)
        ms = self._params[:self._qdim]
        ls = self._params[self._qdim:2 * self._qdim]
        g = self._params[self._qdim * 2]

        Is = 1. / 12. * ms * ls**2

        # TODO(jkg): this can't be the best way to do this
        mass_matrix = q.new_zeros(B, self._qdim, self._qdim)
        for k in range(self._qdim):
            for i in range(self._qdim):
                for j in range(self._qdim):
                    if not (i > k or j > k):
                        # Compute inertia from translational KE terms
                        if i == j:
                            Mij = ls[i]**2 * torch.ones_like(q[:, 0])
                            if i == k:
                                Mij = Mij / 4.
                        else:
                            smaller = i if i < j else j
                            larger = i if i > j else j
                            cosangbet = torch.cos(q[:, smaller + 1:larger + 1].sum(dim=1))
                            Mij = ls[i] * ls[j] * cosangbet

                            if i == k or j == k:
                                Mij = Mij / 2.

                        # Compute inertia from rotational KE terms
                        mass_matrix[:, i, j] += ms[k] * Mij + Is[k]

        return mass_matrix


class MultiLinkAcrobotV(torch.nn.Module):

    def __init__(self, qdim, params=None):
        super().__init__()
        self._qdim = qdim
        if params is None:
            params = torch.abs(torch.randn(self._qdim * 2 + 1))

        assert params.shape == (self._qdim * 2 + 1,)

        if not isinstance(params, torch.nn.Parameter):
            params = torch.nn.Parameter(params)

        self._params = params

    def forward(self, q):
        ms = self._params[:self._qdim]
        ls = self._params[self._qdim:2 * self._qdim]
        g = self._params[self._qdim * 2]

        V_ = ls * torch.cos(q.cumsum(1))
        halfV_ = V_ / 2.0
        V_ = V_.cumsum(1) - halfV_

        V = -torch.sum(ms * g * V_, dim=1, keepdim=True)
        return V


class MultiLinkAcrobot(AbstractRigidBody, torch.nn.Module):

    def __init__(self, qdim, params=None):
        self._qdim = qdim
        self._udim = qdim

        self._thetamask = torch.tensor([1.] * qdim)
        super().__init__()

        if params is None:
            # [m]*qdim, [l]*qdim, g, [tau]*qdim
            params = torch.abs(torch.randn(self._qdim * 2 + 1 + self._qdim))

        if not isinstance(params, torch.nn.Parameter):
            params = torch.nn.Parameter(params)

        assert params.shape == (self._qdim * 2 + 1 + self._qdim,)
        mv_params = params[:self._qdim * 2 + 1]

        self._mass_matrix = MultiLinkAcrobotMM(self._qdim, mv_params)
        self._potential = MultiLinkAcrobotV(self._qdim, mv_params)

        taus = params[self._qdim * 2 + 1:self._qdim * 2 + 1 + self._qdim]
        Bmat = torch.diag_embed(taus.unsqueeze(0))
        self._lin_force = ControlAffineLinearForce(Bmat)

    def mass_matrix(self, q):
        return self._mass_matrix(q)

    def potential(self, q):
        return self._potential(q)

    def generalized_force(self, q, v, u):
        return self._lin_force(q, v, u)

    @property
    def thetamask(self):
        return self._thetamask

    def forward(self, q, v, u=None):
        return self.solve_euler_lagrange(q, v, u)


class DampedMultiLinkAcrobot(MultiLinkAcrobot):

    def __init__(self, qdim, params=None):
        self._qdim = qdim
        if params is None:
            # [m]*qdim, [l]*qdim, g, [tau]*qdim, [eta]*qdim
            params = torch.abs(torch.randn(self._qdim * 2 + 1 + self._qdim * 2))

        if not isinstance(params, torch.nn.Parameter):
            params = torch.nn.Parameter(params)

        assert params.shape == (self._qdim * 2 + 1 + self._qdim * 2,)

        super().__init__(qdim, params=params[:qdim * 2 + 1 + qdim])

        etas = params[qdim * 2 + 1 + qdim:]
        assert etas.shape == (qdim,), etas.shape
        self._visc_force = ViscousJointDampingForce(etas.unsqueeze(0))
        self._generalized_force = GeneralizedForces([self._lin_force, self._visc_force])

    def generalized_force(self, q, v, u):
        return self._generalized_force(q, v, u)
