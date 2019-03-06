#
# File: odesolver.py
#

import abc
import torch


class FixedGridODESolver(metaclass=abc.ABCMeta):

    def __init__(self, func, y0, grid_constructor=None, transforms=None):
        self.func = func
        self.y0 = y0

        if grid_constructor is None:
            grid_constructor = lambda f, y0, t: t

        self.grid_constructor = grid_constructor
        if transforms is None:
            transforms = [lambda x: x for _ in range(len(y0))]

        self.transforms = transforms

    @property
    @abc.abstractmethod
    def order(self):
        """Returns the integration order"""

    @abc.abstractmethod
    def step_func(self, func, t, dt, y, u):
        """Step once through"""

    def integrate(self, t, u=None):
        """
        Arguments:
        - `t`: timesteps to integrate over
        - `u` [list/torch.Tensor]: control inputs list for the time period
        """
        _assert_increasing(t)
        if u is None:
            u = [None] * len(t)

        t = t.type_as(self.y0[0])
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.y0[0])

        solution = [self.y0]
        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self.step_func(self.func, t0, t1 - t0, y0, u=u[j - 1])
            y1 = tuple(trans(y0_ + dy_) for y0_, dy_, trans in zip(y0, dy, self.transforms))
            y0 = y1

            while j < len(t) and t1 >= t[j]:
                solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                j += 1

        return tuple(map(torch.stack, tuple(zip(*solution))))

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        return tuple(dt * f_ for f_ in func(t, y, u=u))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        y_mid = tuple(
            trans(y_ + f_ * dt / 2) for y_, f_, trans in zip(y, func(t, y, u=u), self.transforms))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid, u=u))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        return rk4_alt_step_func(func, t, dt, y, u=u)

    @property
    def order(self):
        return 4


def rk4_alt_step_func(func, t, dt, y, k1=None, u=None):
    """Smaller error with slightly more compute."""
    if k1 is None:
        k1 = func(t, y, u=u)
    k2 = func(t + dt / 3, tuple(y_ + dt * k1_ / 3 for y_, k1_ in zip(y, k1)), u=u)
    k3 = func(t + dt * 2 / 3,
              tuple(y_ + dt * (k1_ / -3 + k2_) for y_, k1_, k2_ in zip(y, k1, k2)), u=u)
    k4 = func(t + dt,
              tuple(y_ + dt * (k1_ - k2_ + k3_) for y_, k1_, k2_, k3_ in zip(y, k1, k2, k3)), u=u)
    return tuple((k1_ + 3 * k2_ + 3 * k3_ + k4_) * (dt / 8)
                 for k1_, k2_, k3_, k4_ in zip(k1, k2, k3, k4))


def odeint(func, y0, t, method=None, transforms=None, **kwargs):
    """Integrates `func` with initial conditions `y0` at points specified by `t`
    Arguments:
    - `func` : function to integrate: ydot = func(t, y, u=u)
    - `y0`   : initial conditions for integration

    Keyword arguments:
    - `method` :  integration scheme in ['euler', 'midpoint', 'rk4'] (default='rk4')
    - `transforms` : a function applied after every step is computed, e.g. wrap_to_pi (default=None)
    """

    tensor_input, func, y0, t = _check_inputs(func, y0, t)
    solver = SOLVERS[method](func, y0, transforms=transforms)
    solution = solver.integrate(t, **kwargs)
    if tensor_input:
        solution = solution[0]
    return solution


class ActuatedODEWrapper:

    def __init__(self, diffeq):
        """
        Wrapper for compat

        Arguments:
        - `diffeq`: torch.nn.Module that takes q, v, u as arguments
        """
        self.diffeq = diffeq

    def forward(self, t, y, u=None):
        (q, v) = y
        qddot = self.diffeq(q, v, u)
        dy = (v, qddot)
        return dy

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)

        return getattr(self.diffeq, attr)


SOLVERS = {
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
}


def _assert_increasing(t):
    assert (t[1:] > t[:-1]).all(), 't must be strictly increasing or decrasing'


def _decreasing(t):
    return (t[1:] < t[:-1]).all()


def _check_inputs(func, y0, t):
    if not isinstance(func, ActuatedODEWrapper):
        func = ActuatedODEWrapper(func)

    tensor_input = False
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = (y0,)

        _base_nontuple_func_ = func
        func = lambda t, y, u: (_base_nontuple_func_(t, y[0], u),)
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'

    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(
            type(y0_))

    for y0_ in y0:
        if not torch.is_floating_point(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.type()))

    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    return tensor_input, func, y0, t
