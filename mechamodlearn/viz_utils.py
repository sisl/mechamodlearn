#
# File: viz_utils.py
#
import matplotlib.pyplot as plt
import torch

from mechamodlearn.odesolver import odeint
from mechamodlearn import utils


def plot_traj(system_x_T_B, model_x_T_B, tstar):
    D = system_x_T_B.shape[-1]
    cm = plt.get_cmap('tab10')
    fig, axs = plt.subplots(1, D, figsize=(6 * D, 6), squeeze=False)
    for d in range(D):  # For each DoF
        for i in range(system_x_T_B.shape[1]):
            axs[0, d].plot(tstar, system_x_T_B[:, i, d], color=cm(i), label='True {}'.format(i),
                           alpha=0.8)
            axs[0, d].plot(tstar, model_x_T_B[:, i, d], color=cm(i), ls='--',
                           label='Pred {}'.format(i), alpha=0.8)

        axs[0, d].set_xlabel('$t$')
        axs[0, d].set_ylabel('$x_{}$'.format(d))
        axs[0, d].legend(frameon=False)

    return fig


def vizqvmodel(model, q_B_T, v_B_T, u_B_T, t_points, method='rk4'):
    B = q_B_T.size(0)
    q_T_B = q_B_T.transpose(1, 0)
    v_T_B = v_B_T.transpose(1, 0)
    u_T_B = u_B_T.transpose(1, 0)
    with torch.no_grad():
        # Simulate forward
        qpreds_T_B, vpreds_T_B = odeint(model, (q_T_B[0],
                                                v_T_B[0]), t_points, u=u_T_B, method=method,
                                        transforms=(lambda x: utils.wrap_to_pi(x, model.thetamask),
                                                    lambda x: x))
        qpreds_T_B = utils.wrap_to_pi(qpreds_T_B.view(-1, model._qdim), model.thetamask).view(
            -1, B, model._qdim)

    q_fig = {
        'qtraj':
            plot_traj(q_T_B.detach().cpu().numpy(),
                      qpreds_T_B.detach().cpu().numpy(), t_points.detach().cpu().numpy())
    }
    v_fig = {
        'vtraj':
            plot_traj(v_T_B.detach().cpu().numpy(),
                      vpreds_T_B.detach().cpu().numpy(), t_points.detach().cpu().numpy())
    }

    return {**q_fig, **v_fig}
