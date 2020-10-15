import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.model import NodeRNN, EdgeRNN
from Model.dataset import MolDataset


def train(train_list, test_list, num_epoch, batch_size, num_layers=2):
    train_dataset = MolDataset(train_list)
    test_dataset = MolDataset(test_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    node_model = NodeRNN(input_size_adj=MAX_TIMESTEP, input_size_node=len(ATOM_IDX), emb_size=64, hidden_lstm_size=128,
                         num_layers=num_layers, hidden_header_size=64, out_features=len(ATOM_IDX),
                         seq_len=MAX_NODE).to(device)
    edge_model = EdgeRNN(input_size=len(BOND_IDX), emb_size=128, hidden_lstm_size=128, num_layers=num_layers,
                         hidden_header_size=64, out_features=len(BOND_IDX), seq_len=MAX_TIMESTEP).to(device)

    node_criterion = nn.CrossEntropyLoss(ignore_index=0)
    edge_criterion = nn.CrossEntropyLoss()

    node_optimizer = optim.Adam(node_model.parameters(), lr=1e-4)
    edge_optimizer = optim.Adam(edge_model.parameters(), lr=1e-4)

    for n in range(num_epoch):
        for i, (S_i, C, S_ij, x_len) in enumerate(train_loader):
            S_i = S_i.to(device)
            C = C.to(device)
            S_ij = S_ij.to(device)
            node_optimizer.zero_grad()
            edge_optimizer.zero_grad()

            y_node = node_model(S_i, C, x_len)

            node_loss = node_criterion(y_node[0].view(-1, len(ATOM_IDX)), C.view(-1, ))
            node_loss.backward(retain_graph=True)
            node_optimizer.step()

            print("EPOCH%d:%d, Train Node loss:%f" % (n, i, node_loss))

            edge_pred = torch.zeros([batch_size, MAX_NODE*MAX_TIMESTEP, len(BOND_IDX)], requires_grad=True).to(device)
            for j in range(MAX_NODE):

                y_edge = edge_model(S_ij[:, j, ], y_node[1][:, j])
                edge_pred[:, MAX_TIMESTEP*j:MAX_TIMESTEP*(j+1), ] = y_edge

            edge_loss = edge_criterion(edge_pred.contiguous().view(-1, len(BOND_IDX)), S_ij.contiguous().view(-1, ))
            edge_loss.backward()
            edge_optimizer.step()

            print("EPOCH%d:%d, Train Edge loss:%f" % (n, i, edge_loss))





if __name__ == "__main__":
    smiles_list = read_smilesset("data/zinc_250k.smi")
    sp = int(len(smiles_list)*0.8)
    train_list = smiles_list[:sp]
    test_list = smiles_list[sp:]

    train(train_list, test_list, num_epoch=10, batch_size=32)
