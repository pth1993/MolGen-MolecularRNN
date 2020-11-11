import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.model import NodeRNN, EdgeRNN
from Model.dataset import MolDataset


def sample_mol(node_model, edge_model):
    with torch.no_grad():
        S_i = torch.zeros([1, MAX_NODE, MAX_TIMESTEP]).to(device)
        C = torch.zeros([1, MAX_NODE], dtype=torch.long).to(device)
        S_ij = torch.zeros([1, MAX_NODE, MAX_TIMESTEP+1], dtype=torch.long).to(device)
        S_ij[:, :, 0] = torch.ones([MAX_NODE])*BOND_IDX["&"]

        A = np.zeros([MAX_NODE, MAX_NODE, NUM_BOND])
        X = np.zeros([MAX_NODE, NUM_ATOM])

        C[0, 0] = ATOM_IDX["&"]
        X[0, ATOM_IDX["&"]] = 1

        i = 0
        while C[0, i] != ATOM_IDX["E"] and i < MAX_NODE-1:
            print(S_i)
            print(S_ij)
            print(C)
            if i == 2:
                time.sleep(10000)
            y_node = node_model(S_i, C, torch.tensor([i+1]).to(device))

            edge_pred = torch.zeros([1, MAX_NODE * MAX_TIMESTEP, len(BOND_IDX)], requires_grad=False).to(device)
            for j in range(i):
                y_edge = edge_model(S_ij[:, j, :-1], y_node[1][:, :, j])
                edge_pred[:, MAX_TIMESTEP * j:MAX_TIMESTEP * (j + 1), ] = y_edge
                print(S_ij[0, j, ])
                print(y_edge)

            node_prob = F.softmax(y_node[0], dim=2).to('cpu').detach().numpy().copy()
            edge_prob = F.softmax(edge_pred, dim=2).to('cpu').detach().numpy().copy()

            node_ind = np.random.choice(range(NUM_ATOM), p=node_prob[0, i, :])
            edge_ind = [np.random.choice(range(NUM_BOND), p=edge_prob[0, j, :]) for j in range(MAX_TIMESTEP)]
            edge_ind_exh = []
            for j in range(len(edge_ind)):
                edge_ind_exh.append(1 if edge_ind[j] != 0 else 0)

            C[0, i+1] = node_ind
            X[i, node_ind] = 1
            if i < MAX_TIMESTEP:
                S_ij[0, i+1, 1:] = torch.tensor(edge_ind)
                S_i[0, i+1, ] = torch.tensor(edge_ind_exh)
                for j in range(i):
                    if edge_ind_exh[j] != 0:
                        A[i, i-j-1, edge_ind_exh[j]] = 1
                        A[i-j-1, i, edge_ind_exh[j]] = 1
            else:
                S_ij[0, i+1, 1:] = torch.tensor(edge_ind)
                S_i[0, i+1, ] = torch.tensor(edge_ind_exh)
                for j in range(MAX_TIMESTEP):
                    if edge_ind_exh[j] != 0:
                        A[i, i-j-1, edge_ind_exh[j]] = 1
                        A[i-j-1, i, edge_ind_exh[j]] = 1

            i += 1

    mol = mat2graph(A, X)

    return mol


def sample_mol_list(node_model, edge_model, num_sample):
    smiles_list = []
    for i in range(num_sample):
        mol = sample_mol(node_model, edge_model)
        if mol is not None:
            smiles_list.append(Chem.MolToSmiles(mol))

    return smiles_list


if __name__ == "__main__":
    node_model = NodeRNN(input_size_adj=MAX_TIMESTEP, input_size_node=len(ATOM_IDX), emb_size=64, hidden_rnn_size=128,
                         hidden_header_size=64, out_features=len(ATOM_IDX), seq_len=MAX_NODE).to(device)
    edge_model = EdgeRNN(input_size=len(BOND_IDX), emb_size=128, hidden_rnn_size=128,
                         hidden_header_size=64, out_features=len(BOND_IDX), seq_len=MAX_TIMESTEP).to(device)
    node_model.load_state_dict(torch.load("data/model/NodeRNN.pth"))
    edge_model.load_state_dict(torch.load("data/model/EdgeRNN.pth"))

    smiles_list = sample_mol_list(node_model, edge_model, num_sample=1)
    # smiles_list = list(set(smiles_list))

    for smiles in smiles_list:
        print(smiles)
    print(len(smiles_list))


