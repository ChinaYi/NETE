import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np

class SingleStageCausalTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageCausalTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualCausalLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageCausalTCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageCausalTCN, self).__init__()
        self.stage1 = SingleStageCausalTCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageCausalTCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class DilatedResidualCausalLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualCausalLayer, self).__init__()
        self.padding = 2 * dilation
        # causal: add padding to the front of the input
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation) #
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.pad(x, [self.padding, 0], 'constant', 0) # add padding to the front of input
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)