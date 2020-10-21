import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.model import NodeRNN, EdgeRNN
from Model.dataset import MolDataset


def sample_mol(node_model, edge_model):
    with torch.no_grad():
        S_i = S_i.to(device)
        C = C.to(device)
        S_ij = S_ij.to(device)

        y_node = node_model(S_i, C, x_len)

        edge_pred = torch.zeros([1, MAX_NODE * MAX_TIMESTEP, len(BOND_IDX)], requires_grad=True).to(device)
        for j in range(max(x_len)):
            y_edge = edge_model(S_ij[:, j, ], y_node[1][:, :, j])
            edge_pred[:, MAX_TIMESTEP * j:MAX_TIMESTEP * (j + 1), ] = y_edge
