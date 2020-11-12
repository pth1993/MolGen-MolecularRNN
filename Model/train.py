import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.model import NodeRNN, EdgeRNN
from Model.dataset import MolDataset


def train(train_list, test_list, num_epoch, batch_size):
    train_dataset = MolDataset(train_list)
    test_dataset = MolDataset(test_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    node_model = NodeRNN(input_size_adj=MAX_TIMESTEP, input_size_node=len(ATOM_IDX), emb_size=64, hidden_rnn_size=256,
                         hidden_header_size=64, out_features=len(ATOM_IDX), seq_len=MAX_NODE).to(device)
    edge_model = EdgeRNN(input_size=len(BOND_IDX), emb_size=16, hidden_rnn_size=128,
                         hidden_header_size=64, out_features=len(BOND_IDX), seq_len=MAX_TIMESTEP).to(device)

    # weights = torch.tensor([1.0, 1000.0, 1000.0, 1000.0, 1.0]).cuda()
    node_criterion = nn.CrossEntropyLoss(ignore_index=0)
    edge_criterion = nn.CrossEntropyLoss()

    node_optimizer = optim.Adam(node_model.parameters(), lr=1e-4)
    edge_optimizer = optim.Adam(edge_model.parameters(), lr=1e-4)

    for n in range(1, num_epoch+1):
        for i, (S_i, C, S_ij, x_len) in enumerate(train_loader):
            S_i = S_i.to(device)
            C = C.to(device)
            S_ij = S_ij.to(device)
            node_optimizer.zero_grad()

            y_node = node_model(S_i, C[:, :-1], x_len)

            edge_pred = torch.zeros([batch_size, MAX_NODE*MAX_TIMESTEP, len(BOND_IDX)], requires_grad=True).to(device)
            edge_true = torch.zeros([batch_size, MAX_NODE*MAX_TIMESTEP, ], dtype=torch.long).to(device)
            for j in range(max(x_len)):
                edge_optimizer.zero_grad()
                y_edge = edge_model(S_ij[:, j, :-1], y_node[1][:, j])
                edge_pred[:, MAX_TIMESTEP*j:MAX_TIMESTEP*(j+1), ] = y_edge
                edge_true[:, MAX_TIMESTEP*j:MAX_TIMESTEP*(j+1), ] = S_ij[:, j, 1:]

            edge_loss = edge_criterion(edge_pred.contiguous().view(-1, len(BOND_IDX)), edge_true.contiguous().view(-1, ))
            edge_loss.backward(retain_graph=True)
            edge_optimizer.step()
            print("EPOCH%d:%d, Train Edge loss:%f" % (n, i, edge_loss))

            node_loss = node_criterion(y_node[0].view(-1, len(ATOM_IDX)), C[:, 1:].contiguous().view(-1, ))
            node_loss.backward()
            node_optimizer.step()
            print("EPOCH%d:%d, Train Node loss:%f" % (n, i, node_loss))

        if n % 10 == 0:
            torch.save(node_model.state_dict(), f"data/model/NodeRNN-ep{n}.pth")
            torch.save(edge_model.state_dict(), f"data/model/EdgeRNN-ep{n}.pth")

        with torch.no_grad():
            test_node_losses = []
            test_edge_losses = []
            for i, (S_i, C, S_ij, x_len) in enumerate(test_loader):
                S_i = S_i.to(device)
                C = C.to(device)
                S_ij = S_ij.to(device)

                y_node = node_model(S_i, C[:, :-1], x_len)

                edge_pred = torch.zeros([batch_size, MAX_NODE * MAX_TIMESTEP, len(BOND_IDX)], requires_grad=False).to(
                    device)
                edge_true = torch.zeros([batch_size, MAX_NODE * MAX_TIMESTEP, ], dtype=torch.long).to(device)
                for j in range(max(x_len)):
                    y_edge = edge_model(S_ij[:, j, :-1], y_node[1][:, j])
                    edge_pred[:, MAX_TIMESTEP * j:MAX_TIMESTEP * (j + 1), ] = y_edge
                    edge_true[:, MAX_TIMESTEP * j:MAX_TIMESTEP * (j + 1), ] = S_ij[:, j, 1:]

                edge_loss = edge_criterion(edge_pred.contiguous().view(-1, len(BOND_IDX)),
                                           edge_true.contiguous().view(-1, ))
                node_loss = node_criterion(y_node[0].view(-1, len(ATOM_IDX)), C[:, 1:].contiguous().view(-1, ))
                test_node_losses.append(float(node_loss))
                test_edge_losses.append(float(edge_loss))

            print("EPOCH%d: Validation Edge loss:%f" % (n, float(np.mean(test_node_losses))))
            print("EPOCH%d: Validation Edge loss:%f" % (n, float(np.mean(test_edge_losses))))


if __name__ == "__main__":
    train_list = read_smilesset("data/zinc_250k_train_sorted.smi")
    test_list = read_smilesset("data/zinc_250k_test_sorted.smi")

    start_time = time.time()
    train(train_list, test_list, num_epoch=50, batch_size=256)
    end_time = time.time()
    print("elapsed time: %f" % (end_time - start_time))
