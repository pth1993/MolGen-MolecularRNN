import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.model import NodeRNN, EdgeRNN


def sample_mol(node_model, edge_model, num_mol, batch_size=512):
    A = np.zeros([(int(num_mol/batch_size)+1)*batch_size, MAX_NODE, MAX_NODE, NUM_BOND])
    X = np.zeros([(int(num_mol/batch_size)+1)*batch_size, MAX_NODE, NUM_ATOM])

    with torch.no_grad():
        for i in range(int(num_mol/batch_size)+1):
            S_i = torch.zeros([batch_size, MAX_NODE, MAX_TIMESTEP]).to(device)
            C = torch.zeros([batch_size, MAX_NODE+1], dtype=torch.long).to(device)
            S_ij = torch.zeros([batch_size, MAX_NODE, MAX_TIMESTEP+1], dtype=torch.long).to(device)
            x_len = [1 for i in range(batch_size)]

            S_ij[:, :, 0] = torch.ones([batch_size, MAX_NODE])*BOND_IDX["&"]
            C[:, 0] = torch.ones([batch_size])*ATOM_IDX["&"]

            j = 0
            while np.sum([C[i, j] for i in range(batch_size)]) > 0 and j < MAX_NODE-1:
                y_node = node_model(S_i, C[:, :-1], x_len)
                y_edge = edge_model(S_ij[:, j, :-1], y_node[1][:, j])
                print(S_ij[:, j, :-1])

                node_prob = F.softmax(y_node[0], dim=2).to('cpu').detach().numpy()
                edge_prob = F.softmax(y_edge, dim=2).to('cpu').detach().numpy()

                node_ind = []
                edge_ind = []
                edge_ind_exh = []
                for k in range(batch_size):
                    nind = np.random.choice(range(NUM_ATOM), p=node_prob[k, j, :])
                    eind = [np.random.choice(range(NUM_BOND), p=edge_prob[0, l, :]) for l in range(MAX_TIMESTEP)]
                    eind_exh = [1 if eind[l] != 0 else 0 for l in range(MAX_TIMESTEP)]

                    node_ind.append(nind)
                    edge_ind.append(eind)
                    edge_ind_exh.append(eind_exh)
                print("=====================")

                for k in range(batch_size):
                    C[k, j+1] = node_ind[k]
                    X[k, j, node_ind[k]] = 1
                    for l in range(MAX_TIMESTEP):
                        if edge_ind_exh[k][l] != 0 and j-l > 0:
                            S_ij[k, j+1, l+1] = edge_ind[k][l]
                            S_i[k, j+1, l] = edge_ind_exh[k][l]
                            A[k, j, j-l-1, edge_ind[k][l]] = 1
                            A[k, j-l-1, j, edge_ind[k][l]] = 1
                    x_len[k] += 1
                j += 1

    smiles_list = []
    for i in range(num_mol):
        mol = mat2graph(A[i], X[i])
        if mol is not None:
            smiles_list.append(Chem.MolToSmiles(mol))
    smiles_list = list(set(smiles_list))

    return smiles_list


if __name__ == "__main__":
    node_model = NodeRNN(input_size_adj=MAX_TIMESTEP, input_size_node=len(ATOM_IDX), emb_size=64, hidden_rnn_size=256,
                         hidden_header_size=64, out_features=len(ATOM_IDX), seq_len=MAX_NODE).to(device)
    edge_model = EdgeRNN(input_size=len(BOND_IDX), emb_size=16, hidden_rnn_size=128,
                         hidden_header_size=64, out_features=len(BOND_IDX), seq_len=MAX_TIMESTEP).to(device)
    node_model.load_state_dict(torch.load("data/model/NodeRNN-ep20.pth"))
    edge_model.load_state_dict(torch.load("data/model/EdgeRNN-ep20.pth"))

    smiles_list = sample_mol(node_model, edge_model, num_mol=1, batch_size=1)
    # smiles_list = list(set(smiles_list))

    for smiles in smiles_list:
        print(smiles)
    print(len(smiles_list))


