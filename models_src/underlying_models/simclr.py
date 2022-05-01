##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class SimCLRModel(nn.Module):
    """ This class implements a self-supervised image processing model, described in the paper
        "A Simple Framework for Contrastive Learning of Visual Representations", by Ting Chen, Simon Kornblith,
        Mohammad Norouzi, Geoffrey Hinton.
    """

    def __init__(self, output_encoder=True, feature_dim=128):
        super(SimCLRModel, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        self.output_encoder = output_encoder

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        if self.output_encoder:
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        else:
            return F.normalize(out, dim=-1)

    """ Clean model from unnecessary parameters, created by the thop package in the original SimCLR repo. """

    @staticmethod
    def clean_state_dict(messy_state_dict):
        cleaned_state_dict = {}
        for n, p in messy_state_dict.items():
            if "total_ops" not in n and "total_params" not in n:
                cleaned_state_dict[n] = p

        return cleaned_state_dict

    """ Change size of projection layer in an existing state dict. """

    @staticmethod
    def adjust_projection_in_state_dict(state_dict, output_size):
        new_state_dict = {}
        for n, p in state_dict.items():
            if not n.startswith('g.'):
                new_state_dict[n] = p

        dummy_layer = nn.Linear(2048, output_size)
        new_state_dict['g.weight'] = dummy_layer.weight
        new_state_dict['g.bias'] = dummy_layer.bias

        return new_state_dict
