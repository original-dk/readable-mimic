"""Muon for hidden matrices, Adam for boundary layers and vectors."""
import torch
from torch import nn


class MuonAdam:
    def __init__(self, model, networks, learning_rate, betas):
        hidden_ids = set()
        for network in networks:
            layers = [layer for layer in network.modules() if isinstance(layer, nn.Linear)]
            hidden_ids.update(id(layer.weight) for layer in layers[1:-1])
        matrices = [parameter for parameter in model.parameters() if id(parameter) in hidden_ids]
        remaining = [parameter for parameter in model.parameters() if id(parameter) not in hidden_ids]
        self.muon = torch.optim.Muon(matrices, lr=learning_rate, weight_decay=0.0,
                                     adjust_lr_fn='match_rms_adamw')
        self.adam = torch.optim.Adam(remaining, lr=learning_rate, betas=betas)

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {'muon': self.muon.state_dict(), 'adam': self.adam.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state['muon'])
        self.adam.load_state_dict(state['adam'])


def make_optimizer(name, model, networks, learning_rate, betas):
    factories = {
        'adam': lambda: torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas),
        'muon': lambda: MuonAdam(model, networks, learning_rate, betas),
    }
    return factories[name]()
