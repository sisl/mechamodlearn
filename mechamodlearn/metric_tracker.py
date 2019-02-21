# File: metric_tracker.py
#
# Based on https://github.com/allenai/allennlp/blob/master/allennlp/training/metric_tracker.py
from typing import Optional, Dict, Any, Iterable


class MetricTracker:

    def __init__(self, patience: Optional[int] = None, should_decrease: bool = None):
        self._best_so_far: Optional[float] = None
        self._patience = patience
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self.best_epoch_metrics: Dict[str, float] = {}
        self._epoch_number = 0
        self.best_epoch: Optional[int] = None

        self._should_decrease = should_decrease

    def clear(self) -> None:
        """
        Clears out the tracked metrics, but keeps the patience and should_decrease settings.
        """
        self._best_so_far = None
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self._epoch_number = 0
        self.best_epoch = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_so_far": self._best_so_far,
            "patience": self._patience,
            "epochs_with_no_improvement": self._epochs_with_no_improvement,
            "is_best_so_far": self._is_best_so_far,
            "should_decrease": self._should_decrease,
            "best_epoch_metrics": self.best_epoch_metrics,
            "epoch_number": self._epoch_number,
            "best_epoch": self.best_epoch
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        A ``Trainer`` can use this to hydrate a metric tracker from a serialized state.
        """
        self._best_so_far = state_dict["best_so_far"]
        self._patience = state_dict["patience"]
        self._epochs_with_no_improvement = state_dict["epochs_with_no_improvement"]
        self._is_best_so_far = state_dict["is_best_so_far"]
        self._should_decrease = state_dict["should_decrease"]
        self.best_epoch_metrics = state_dict["best_epoch_metrics"]
        self._epoch_number = state_dict["epoch_number"]
        self.best_epoch = state_dict["best_epoch"]

    def add_metric(self, metric: float) -> None:
        """
        Record a new value of the metric and update the various things that depend on it.
        """
        new_best = ((self._best_so_far is None) or
                    (self._should_decrease and metric < self._best_so_far) or
                    (not self._should_decrease and metric > self._best_so_far))

        if new_best:
            self.best_epoch = self._epoch_number
            self._is_best_so_far = True
            self._best_so_far = metric
            self._epochs_with_no_improvement = 0
        else:
            self._is_best_so_far = False
            self._epochs_with_no_improvement += 1
        self._epoch_number += 1

    def add_metrics(self, metrics: Iterable[float]) -> None:
        """
        Helper to add multiple metrics at once.
        """
        for metric in metrics:
            self.add_metric(metric)

    def is_best_so_far(self) -> bool:
        """
        Returns true if the most recent value of the metric is the best so far.
        """
        return self._is_best_so_far

    def should_stop_early(self) -> bool:
        """
        Returns true if improvement has stopped for long enough.
        """
        if self._patience is None:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience
