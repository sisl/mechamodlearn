#
# File: trainer.py
#
from typing import Dict, List, Optional, Tuple

import abc
import datetime
from pathlib import Path
import re
import os
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from mechamodlearn import dataset, logger, nested, transform, utils
from mechamodlearn.metric_tracker import MetricTracker
from mechamodlearn.odesolver import ActuatedODEWrapper, odeint


class TrainerBase(abc.ABC):

    def __init__(self, model, optimizer, logdir, metric_tracker):
        self.model = model
        self.optimizer = optimizer
        self._logdir = logdir
        # Track is_best_so_far and early stopping
        self._metric_tracker = metric_tracker
        logger.setup(self._logdir, action='k')
        self._setup_debug()

    def _setup_debug(self):
        import sys
        old_hook = sys.excepthook

        def new_hook(typ, value, tb):
            old_hook(typ, value, tb)
            if typ != KeyboardInterrupt:
                import ipdb
                ipdb.post_mortem(tb)

        sys.excepthook = new_hook

    def _parameter_and_gradient_statistics(self) -> None:
        for name, param in self.model.named_parameters():
            logger.logkv("parameter_mean/" + name, param.data.mean().item())
            logger.logkv("parameter_std/" + name, param.data.std().item())
            logger.logkv("parameter_norm/" + name, param.data.norm().item())

            if param.grad is not None:
                grad_data = param.grad.data

                # skip empty gradients
                if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                    logger.logkv("gradient_mean/" + name, grad_data.mean().item())
                    logger.logkv("gradient_std/" + name, grad_data.std().item())
                    logger.logkv("gradient_norm/" + name, grad_data.norm().item())
                else:
                    logger.info("No gradient for {}, skipping.".format(name))

    def _save_checkpoint(self, epoch: int) -> None:
        model_path = Path(self._logdir) / "model_state_epoch_{}.th".format(epoch)
        model_state = self.model.state_dict()
        torch.save(model_state, model_path)

        training_state = {
            'epoch': epoch,
            'metric_tracker': self._metric_tracker.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        training_path = Path(self._logdir) / "training_state_epoch_{}.th".format(epoch)
        torch.save(training_state, training_path)

        if self._metric_tracker.is_best_so_far():
            logger.info("Best validation performance so far. Copying weights to {}/best.th".format(
                self._logdir))
            shutil.copyfile(model_path, Path(self._logdir) / "best.th")

    def _restore_checkpoint(self, optstate=True) -> int:
        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            return 0

        model_path, training_state_path = latest_checkpoint

        model_state = torch.load(model_path, map_location='cpu')
        training_state = torch.load(training_state_path, map_location='cpu')
        self.model.load_state_dict(model_state)
        if optstate:
            self.optimizer.load_state_dict(training_state["optimizer"])

        move_optimizer_to_gpu(self.optimizer)
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        return epoch_to_return

    def find_latest_checkpoint(self):
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        have_checkpoint = (self._logdir is not None and any("model_state_epoch_" in x
                                                            for x in os.listdir(self._logdir)))

        if not have_checkpoint:
            return None

        serialization_files = os.listdir(self._logdir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
            # pylint: disable=anomalous-backslash-in-string
            re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1) for x in model_checkpoints
        ]
        int_epochs = []
        for epoch in found_epochs:
            pieces = epoch.split('.')
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), 0])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])

        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == 0:
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        model_path = Path(self._logdir) / "model_state_epoch_{}.th".format(epoch_to_load)
        training_state_path = Path(
            self._logdir) / "training_state_epoch_{}.th".format(epoch_to_load)

        return (model_path, training_state_path)

    @abc.abstractmethod
    def train(self):
        """Main train"""


class OfflineTrainer(TrainerBase):

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            dt: float,
            train_dataset,
            validation_dataset,
            patience: Optional[int] = None,
            shuffle: bool = True,
            pred_horizon: int = 3,
            num_epochs: int = 20,
            batch_size: int = 64,
            vlambda: float = 1.0,
            logdir: Optional[str] = None,
            learning_rate_scheduler=None,
            summary_interval: int = 10,
            ckpt_interval: int = 10,
            integration_method: str = 'rk4',
            should_log_parameter_statistics: bool = False,
            log_viz: bool = False,
            viz_func=None,
            device='cpu',):

        metric_tracker = MetricTracker(patience, True)
        super().__init__(model, optimizer, logdir, metric_tracker)

        self._dt = dt
        self._pred_horizon = pred_horizon
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset

        self._shuffle = shuffle
        self._num_epochs = num_epochs
        self._batch_size = batch_size

        self._learning_rate_scheduler = learning_rate_scheduler
        self._summary_interval = summary_interval
        self._ckpt_interval = ckpt_interval

        self._vlambda = vlambda
        self._integration_method = integration_method
        self._should_log_parameter_statistics = should_log_parameter_statistics
        self._log_viz = log_viz
        self._viz_func = viz_func
        self._device = device
        self.model.to(device=self._device)

    def train(self):
        logger.info("Begin training...")
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise Exception(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing log "
                "directory?")

        train_metrics = {}
        valid_metrics = {}
        metrics = {}
        this_epoch_valid_metric: float = None
        epochs_trained = 0
        training_start_time = time.time()

        metrics['best_epoch'] = self._metric_tracker.best_epoch

        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation/" + key] = value

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            with utils.Timer() as tr_dt:
                train_metrics = self._train_epoch(epoch)

            train_metrics['epoch_time'] = tr_dt.dt

            # get peak of memory usage
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(
                    metrics.get('peak_cpu_memory_MB', 0), train_metrics['cpu_memory_MB'])

            if self._validation_dataset is not None:
                with utils.Timer() as val_dt:
                    valid_metrics = self._validation()
                    this_epoch_valid_metric = valid_metrics['loss/mean']

                    self._metric_tracker.add_metric(this_epoch_valid_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

                valid_metrics['epoch_time'] = val_dt.dt

            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime("%H:%M:%S",
                                                         time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for k, v in train_metrics.items():
                logger.logkv("training/{}".format(k), v)

            for k, v in valid_metrics.items():
                logger.logkv("validation/{}".format(k), v)

            if self._logdir:
                if (epochs_trained % self._ckpt_interval == 0) or (
                        epochs_trained + 1) == self._num_epochs:
                    self._save_checkpoint(epoch)

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in valid_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = valid_metrics

            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step()

            if (epochs_trained == 0) or epochs_trained % self._summary_interval == 0:
                if self._should_log_parameter_statistics:
                    self._parameter_and_gradient_statistics()

                train_metrics_ = self._metrics(self._train_dataset)
                for k, v in train_metrics_.items():
                    logger.logkv("training/{}".format(k), v)

                val_metrics_ = self._metrics(self._validation_dataset)
                for k, v in val_metrics_.items():
                    logger.logkv("validation/{}".format(k), v)

                if self._log_viz:
                    try:
                        fig_map = self._viz_func(self.model)
                        for k, fig in fig_map.items():
                            logger.add_figure(k, fig)
                    except Exception as e:
                        logger.info("Couldn't log viz: {}".format(e))

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s",
                        time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            logger.dumpkvs()
            epochs_trained += 1

        return metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        logger.info("Epoch {}/{}".format(epoch, self._num_epochs - 1))
        peak_mem_usage = utils.peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_mem_usage}")

        loss_ls = []
        loss_info_ls = []
        losstimer_ls = []
        gradtimer_ls = []

        train_data = self._train_dataset
        if isinstance(train_data, dataset.ActuatedTrajectoryDataset):
            train_data = transform.odepred_transform(train_data, self._pred_horizon)

        train_generator = torch.utils.data.DataLoader(train_data, shuffle=self._shuffle,
                                                      batch_size=self._batch_size)
        for qs, vs, us in tqdm.tqdm(train_generator, total=len(train_generator)):
            self.optimizer.zero_grad()
            with utils.Timer() as losstime:
                loss, loss_info = compute_qvloss(
                    ActuatedODEWrapper(self.model),
                    torch.stack(qs).to(self._device),
                    torch.stack(vs).to(self._device),
                    torch.stack(us).to(self._device), dt=self._dt, vlambda=self._vlambda,
                    method=self._integration_method)

            with utils.Timer() as gradtime:
                loss.backward()

            self.optimizer.step()

            loss_ls.append(loss.cpu().detach().numpy())
            loss_info_ls.append(loss_info)
            losstimer_ls.append(losstime.dt)
            gradtimer_ls.append(gradtime.dt)

        metrics = {}
        loss_info = nested.zip(*loss_info_ls)
        metrics['cpu_memory_MB'] = peak_mem_usage
        metrics['loss/mean'] = np.mean(loss_ls)
        metrics['loss/std'] = np.std(loss_ls)
        metrics['log10loss/mean'] = np.mean(np.log10(loss_ls))
        metrics['log10loss/std'] = np.std(np.log10(loss_ls))
        for k, val in loss_info.items():
            metrics['loss/{}/mean'.format(k)] = np.mean(val)
            metrics['loss/{}/std'.format(k)] = np.std(val)

        metrics['time/loss/mean'] = np.mean(losstimer_ls)
        metrics['time/loss/std'] = np.std(losstimer_ls)
        metrics['time/loss/max'] = np.max(losstimer_ls)
        metrics['time/loss/min'] = np.min(losstimer_ls)
        metrics['time/grad/mean'] = np.mean(gradtimer_ls)
        metrics['time/grad/std'] = np.std(gradtimer_ls)
        metrics['time/grad/max'] = np.max(gradtimer_ls)
        metrics['time/grad/min'] = np.min(gradtimer_ls)

        if self._learning_rate_scheduler:
            metrics['lr'] = self._learning_rate_scheduler.get_lr()[0]

        return metrics

    def _validation(self) -> Dict[str, float]:
        logger.info("Validating")
        val_data = self._validation_dataset
        if isinstance(val_data, dataset.ActuatedTrajectoryDataset):
            val_data = transform.odepred_transform(val_data, self._pred_horizon)

        val_generator = torch.utils.data.DataLoader(val_data, shuffle=False,
                                                    batch_size=self._batch_size)
        loss_ls = []
        loss_info_ls = []
        for qs, vs, us in tqdm.tqdm(val_generator, total=len(val_generator)):
            with torch.no_grad():
                loss, loss_info = compute_qvloss(
                    ActuatedODEWrapper(self.model),
                    torch.stack(qs).to(self._device),
                    torch.stack(vs).to(self._device),
                    torch.stack(us).to(self._device), dt=self._dt, vlambda=self._vlambda,
                    method=self._integration_method)

            loss_ls.append(loss.cpu().detach().item())
            loss_info_ls.append(loss_info)

        metrics = {}
        loss_info = nested.zip(*loss_info_ls)

        metrics['loss/mean'] = np.mean(loss_ls)
        metrics['loss/std'] = np.std(loss_ls)
        metrics['log10loss/mean'] = np.mean(np.log10(loss_ls))
        metrics['log10loss/std'] = np.std(np.log10(loss_ls))
        for k, val in loss_info.items():
            metrics['loss/{}/mean'.format(k)] = np.mean(val)
            metrics['loss/{}/std'.format(k)] = np.std(val)

        return metrics

    def _metrics(self, data) -> Dict[str, float]:
        if not isinstance(data, dataset.ActuatedTrajectoryDataset):
            return {}

        B = data.q_B_T.size(0)
        T = data.q_B_T.size(1)
        q0 = data.q_B_T[:, 0]
        v0 = data.v_B_T[:, 0]
        with torch.no_grad():
            loss, info, (qpreds_T_B, vpreds_T_B) = compute_qvloss(
                ActuatedODEWrapper(self.model),
                data.q_B_T.transpose(1, 0).to(self._device),
                data.v_B_T.transpose(1, 0).to(self._device),
                data.u_B_T.transpose(1, 0).to(self._device), dt=self._dt, vlambda=self._vlambda,
                method=self._integration_method, preds=True)

        qpreds_B_T = qpreds_T_B.transpose(1, 0)
        vpreds_B_T = vpreds_T_B.transpose(1, 0)
        q_mse = F.mse_loss(data.q_B_T.to(device=qpreds_B_T.device), qpreds_B_T)
        v_mse = F.mse_loss(data.v_B_T.to(device=vpreds_B_T.device), vpreds_B_T)

        qdiff = utils.diffangles(
            data.q_B_T.reshape(-1, self.model._qdim).to(device=qpreds_B_T.device),
            qpreds_B_T.reshape(-1, self.model._qdim)).reshape(B, T, self.model._qdim).cpu().numpy()
        vdiff = (data.v_B_T.to(device=vpreds_B_T.device) - vpreds_B_T).cpu().numpy()
        q_ts_error_B_D = np.apply_along_axis(lambda x: utils.time_series_norm(x, self._dt), 1,
                                             qdiff)

        assert q_ts_error_B_D.shape[0] == B
        v_ts_error_B_D = np.apply_along_axis(lambda x: utils.time_series_norm(x, self._dt), 1,
                                             vdiff)
        assert v_ts_error_B_D.shape[0] == B

        q_ts_error = {
            'tsnorm_q_{}'.format(d): q_ts_error_B_D[:, d].mean()
            for d in range(q_ts_error_B_D.shape[-1])
        }
        v_ts_error = {
            'tsnorm_v_{}'.format(d): v_ts_error_B_D[:, d].mean()
            for d in range(v_ts_error_B_D.shape[-1])
        }

        metrics = {
            'metric_q_loss': info['q_loss'],
            'metric_v_loss': info['v_loss'],
            'metric_q_log10loss': np.log10(info['q_loss']),
            'metric_v_log10loss': np.log10(info['v_loss']),
            'metric_loss': loss.item(),
            'metric_log10loss': np.log10(loss.item()),
            'metric_q_mse': q_mse.item(),
            'metric_v_mse': v_mse.item(),
            **q_ts_error,
            **v_ts_error,
        }
        return metrics


def compute_qvloss(model, q_T_B, v_T_B, u_T_B, dt, vlambda=1.0, method='rk4', preds=False):
    T = u_T_B.size(0)
    t_points = torch.arange(0, T * dt, dt).to(q_T_B.device).requires_grad_(True)
    assert len(t_points) == T
    # Simulate forward
    qpreds_T_B, vpreds_T_B = odeint(model, (q_T_B[0], v_T_B[0]), t_points, u=u_T_B, method=method,
                                    transforms=(lambda x: utils.wrap_to_pi(x, model.thetamask),
                                                lambda x: x))

    # Wrap angles
    qpreds_T_B = utils.wrap_to_pi(qpreds_T_B.view(-1, model._qdim), model.thetamask).view(
        T, -1, model._qdim)

    qdiff = utils.diffangles(
        q_T_B.view(-1, model._qdim), qpreds_T_B.view(-1, model._qdim), mask=model.thetamask).view(
            T, -1, model._qdim)
    q_loss = 0.5 * (qdiff**2).sum(0).sum(-1).mean()

    vdiff = (v_T_B - vpreds_T_B)
    v_loss = 0.5 * (vdiff**2).sum(0).sum(-1).mean()

    loss = q_loss + vlambda * v_loss
    info = dict(q_loss=q_loss.cpu().detach().item(), v_loss=v_loss.cpu().detach().item())

    if preds:
        return loss, info, (qpreds_T_B, vpreds_T_B)

    return loss, info


def move_optimizer_to_gpu(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if isinstance(param_state[k], torch.Tensor):
                        param_state[k] = param_state[k].cuda(device=param.get_device())
