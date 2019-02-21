import torch

from .pendulum import ActuatedDampedPendulum, ActuatedSimplePendulum
from .mlacrobot import MultiLinkAcrobot, DampedMultiLinkAcrobot

DEFAULT_SYS_PARAMS = {
    'simplependulum': torch.tensor([1.0, 1.0, 1.0]),
    'dampedpendulum': torch.tensor([1.0, 10.0, 1.0, -0.5]),
    '2linkdampedacrobot': torch.tensor([10.] * 2 + [1.] * 2 + [10.] + [1.] * 2 + [-0.5] * 2)
}
