# File: models.py
#
import torch

from mechamodlearn import nn, utils


class SharedMMVEmbed(torch.nn.Module):

    def __init__(self, qdim, hidden_sizes):
        self._qdim = qdim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._lnet = nn.LNMLP(qdim, hidden_sizes[:-1], hidden_sizes[-1])

    def forward(self, q):
        embed = self._lnet(q)
        return embed


class CholeskyMMNet(torch.nn.Module):

    def __init__(self, qdim, embed=None, hidden_sizes=None, bias=2.0, pos_enforce=lambda x: x):
        self._qdim = qdim
        self._bias = bias
        self._pos_enforce = pos_enforce
        super().__init__()
        if embed is None:
            if hidden_sizes is None:
                raise ValueError("embed and hidden_sizes; both can't be None")
            embed = SharedMMVEmbed(qdim, hidden_sizes)

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], int(qdim * (qdim + 1) / 2))

    def forward(self, q):
        B = q.size(0)
        if self._qdim > 1:
            L_params = self.out(self.embed(q))

            L_diag = self._pos_enforce(L_params[:, :self._qdim])
            L_diag += self._bias
            L_tril = L_params[:, self._qdim:]
            L = q.new_zeros(B, self._qdim, self._qdim)
            L = utils.bfill_lowertriangle(L, L_tril)
            L = utils.bfill_diagonal(L, L_diag)
            M = L @ L.transpose(-2, -1)

        else:
            M = self._pos_enforce((self.out(self.embed(q)) + self._bias).unsqueeze(1))

        return M


class PotentialNet(torch.nn.Module):

    def __init__(self, qdim, embed=None, hidden_sizes=None):
        self._qdim = qdim
        super().__init__()
        if embed is None:
            if hidden_sizes is None:
                raise ValueError("embed and hidden_sizes; both can't be None")

            embed = SharedMMVEmbed(qdim, hidden_sizes)

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, q):
        return self.out(self.embed(q))


class GeneralizedForceNet(torch.nn.Module):

    def __init__(self, qdim, udim, hidden_sizes):
        self._qdim = qdim
        self._udim = udim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._net = nn.LNMLP(self._qdim * 2 + self._udim, hidden_sizes, qdim)

    def forward(self, q, v, u):
        B = q.size(0)
        x = torch.cat([q, v, u], dim=-1)
        F = self._net(x)
        F = F.unsqueeze(2)
        assert F.shape == (B, self._qdim, 1), F.shape
        return F


class ControlAffineForceNet(torch.nn.Module):

    def __init__(self, qdim, udim, hidden_sizes):
        self._qdim = qdim
        self._udim = udim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._net = nn.LNMLP(self._qdim, hidden_sizes, qdim * udim)

    def forward(self, q, v, u):
        B = q.size(0)
        Bmat = self._net(q).view(B, self._qdim, self._udim)
        F = Bmat @ u.unsqueeze(2)
        assert F.shape == (B, self._qdim, 1), F.shape
        return F


class ControlAffineLinearForce(torch.nn.Module):

    def __init__(self, B):
        """
        B needs to be shaped (1, qdim, qdim) usually diagonal
        """
        super().__init__()
        if not isinstance(B, torch.nn.Parameter):
            B = torch.nn.Parameter(B)

        self._B = B

    def forward(self, q, v, u):
        N = q.size(0)
        assert u.size(0) == N
        assert self._B.shape == (1, q.size(1), q.size(1)), self._B.shape
        B = self._B
        F = B @ u.unsqueeze(2)
        assert F.shape == (N, q.size(1), 1), F.shape
        return F


class ViscousJointDampingForce(torch.nn.Module):

    def __init__(self, eta):
        """
        eta needs to be shaped (1, qdim)
        """
        super().__init__()
        if not isinstance(eta, torch.nn.Parameter):
            eta = torch.nn.Parameter(eta)

        self._eta = eta

    def forward(self, q, v, u):
        N = q.size(0)
        assert self._eta.size(1) == v.size(1)
        F = (self._eta * v).unsqueeze(2)
        assert F.shape == (N, q.size(1), 1), F.shape
        return F


class GeneralizedForces(torch.nn.Module):

    def __init__(self, forces):
        super().__init__()
        self.forces = torch.nn.ModuleList(forces)

    def forward(self, q, v, u):
        F = torch.zeros(q.size(0), q.size(1), 1)
        for f in self.forces:
            F += f(q, v, u)

        return F
