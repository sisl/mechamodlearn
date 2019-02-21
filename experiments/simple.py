#!/usr/bin/env python3
#
# File: simple.py
#
from datetime import datetime
from pathlib import Path

import click
import torch

from mechamodlearn import dataset, utils, viz_utils
from mechamodlearn.trainer import OfflineTrainer
from mechamodlearn.systems import ActuatedDampedPendulum, DampedMultiLinkAcrobot, DEFAULT_SYS_PARAMS
from mechamodlearn.rigidbody import LearnedRigidBody

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

SYSTEMS = {
    'dampedpendulum': ActuatedDampedPendulum,
    '2linkdampedacrobot': lambda p: DampedMultiLinkAcrobot(2, p),
}


def get_dataset(system, T: float, dt: float, ntrajs: int, uscale: float, qrange=(-1, 1),
                vrange=(-10, 10)):

    t_points = torch.arange(0, T, dt).requires_grad_(True)
    q0 = torch.stack([torch.empty(system._qdim).uniform_(*qrange)
                      for _ in range(ntrajs)]).requires_grad_(True)
    v0 = torch.stack([torch.empty(system._qdim).uniform_(*vrange)
                      for _ in range(ntrajs)]).requires_grad_(True)
    u_T_B = torch.randn(len(t_points), ntrajs, system._udim) * uscale

    data = dataset.ActuatedTrajectoryDataset.FromSystem(system, q0, v0, u_T_B, t_points)
    return data


def train(seed: int, dt: float, system: str, pred_horizon: int, num_epochs: int, batch_size: int,
          lr: float, ntrajs: int, uscale: float, logdir: str):
    args = locals()
    args.pop('logdir')
    args.pop('num_epochs')
    exp_name = ",".join(["=".join([key, str(val)]) for key, val in args.items()])

    utils.set_rng_seed(seed)

    system = SYSTEMS[system](DEFAULT_SYS_PARAMS[system])

    train_dataset = get_dataset(system, pred_horizon * dt, dt, ntrajs, uscale)
    valid_dataset = get_dataset(system, pred_horizon * dt, dt, ntrajs, uscale)
    test_dataset = get_dataset(system, 4, dt, 4, uscale)

    def viz(model):
        t_points = torch.arange(0, test_dataset.q_B_T.size(1) * dt, dt).to(device=DEVICE)
        return viz_utils.vizqvmodel(model,
                                    test_dataset.q_B_T.to(device=DEVICE),
                                    test_dataset.v_B_T.to(device=DEVICE),
                                    test_dataset.u_B_T.to(device=DEVICE), t_points)

    model = LearnedRigidBody(system._qdim, system._udim, system.thetamask,
                             hidden_sizes=[32, 32, 32, 32])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if logdir is not None:
        logdir = Path(logdir) / exp_name

    trainer = OfflineTrainer(model, opt, dt, train_dataset, valid_dataset, num_epochs=num_epochs,
                             batch_size=batch_size, log_viz=True, viz_func=viz, ckpt_interval=100,
                             summary_interval=200, shuffle=True, logdir=logdir, device=DEVICE)

    metrics = trainer.train()

    if logdir is not None:
        torch.save(metrics, Path(logdir) / 'metrics_{:%Y%m%d-%H%M%S}.pt'.format(datetime.now()))
    return metrics


@click.command()
@click.option('--seed', default=42, type=int)
@click.option('--dt', default=0.01, type=float)
@click.option('--system', default='dampedpendulum', type=str)
@click.option('--pred-horizon', default=3, type=int)
@click.option('--num-epochs', default=1000, type=int)
@click.option('--batch-size', default=128, type=int)
@click.option('--lr', default=3e-4, type=float)
@click.option('--ntrajs', default=8192, type=int)
@click.option('--uscale', default=30.0, type=float)
@click.option('--logdir', default=None, type=str)
def run(seed, dt, system, pred_horizon, num_epochs, batch_size, lr, ntrajs, uscale, logdir):
    metrics = train(seed, dt, system, pred_horizon, num_epochs, batch_size, lr, ntrajs, uscale,
                    logdir)
    print(metrics)


if __name__ == '__main__':
    run()
