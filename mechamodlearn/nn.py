# File: nn.py
#
import torch


class Identity(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


ACTIVATIONS = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'elu': torch.nn.ELU,
    'identity': Identity
}


class LNMLP(torch.nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, activation='tanh', gain=1.0,
                 ln=False):
        self._hidden_sizes = hidden_sizes
        self._gain = gain
        self._ln = ln
        super().__init__()
        activation = ACTIVATIONS[activation]
        layers = [torch.nn.Linear(input_size, hidden_sizes[0])]
        layers.append(activation())
        if ln:
            layers.append(torch.nn.LayerNorm(hidden_sizes[0]))

        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation())
            if ln:
                layers.append(torch.nn.LayerNorm(hidden_sizes[i + 1]))

        layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))

        self._layers = layers
        self.mlp = torch.nn.Sequential(*layers)
        self.reset_params(gain=gain)

    def forward(self, inp):
        return self.mlp(inp)

    def reset_params(self, gain=1.0):
        self.apply(lambda x: weights_init_mlp(x, gain=gain))


def weights_init_mlp(m, gain=1.0):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init_normc_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0)


def init_normc_(weight, gain=1.0):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
