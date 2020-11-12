import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.model import NodeRNN, EdgeRNN


class MolDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, max_dim=12, seq_len=40):
        super(MolDataset, self).__init__()
        self.smiles_list = smiles_list
        self.max_dim = max_dim
        self.seq_len = seq_len

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, item):
        smiles = self.smiles_list[item]
        mol = Chem.MolFromSmiles(smiles)

        # convert BFS ordering
        A, X = Smiles2Graph(mol, is_Tensor=False)
        mol = orderBFSmol(A, X, num_atom=mol.GetNumAtoms())
        A, X = Smiles2Graph(mol, is_Tensor=False)
        A_exs = np.where(A > 0, 1, 0)

        # create Input for NodeRNN
        S_i = np.zeros([self.seq_len, self.max_dim])
        for i in range(A.shape[0]):
            if i < self.max_dim:
                S_i[i, :i] = A_exs[i, :i]
            else:
                S_i[i, ] = A_exs[i, i-12:i][::-1]
        C = np.argmax(X, axis=1)
        C = np.insert(C, 0, ATOM_IDX["&"])
        atom_num = mol.GetNumAtoms()

        # create Input for EdgeRNN
        S_ij = np.zeros([self.seq_len, self.max_dim+1])
        S_ij[:, 0] = np.ones(self.seq_len) * BOND_IDX["&"]
        for i in range(A.shape[0]):
            if i < self.max_dim:
                S_ij[i, 1:i+1] = A[i, :i]
            else:
                S_ij[i, 1:] = A[i, i-12:i][::-1]

        # convert numpy.ndarray to torch.tensor
        S_i = torch.tensor(S_i, dtype=torch.float)
        S_ij = torch.tensor(S_ij, dtype=torch.long)
        C = torch.tensor(C, dtype=torch.long)

        return S_i, C, S_ij, atom_num


if __name__ == "__main__":
    smiles_list = read_smilesset("data/zinc_250k.smi")
    train_dataset = MolDataset(smiles_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    nodemodel = NodeRNN(input_size_adj=MAX_TIMESTEP, input_size_node=len(ATOM_IDX), emb_size=128, hidden_rnn_size=256,
                        hidden_header_size=128, out_features=len(ATOM_IDX), seq_len=40)
    edgemodel = EdgeRNN(input_size=len(BOND_IDX), emb_size=16, hidden_rnn_size=128, hidden_header_size=64,
                        out_features=len(BOND_IDX), seq_len=MAX_TIMESTEP)
    nodemodel.to(device)
    edgemodel.to(device)

    for i, (S_i, C, S_ij, x_len) in enumerate(train_loader):
        print(S_i.shape)
        print(S_ij.shape, S_ij[:, :, :MAX_TIMESTEP].shape)
        print(C.shape, C[:, :-1].shape)
        print("---------------------")
        S_i = S_i.to(device)
        C = C.to(device)
        y = nodemodel(S_i, C[:, :-1].contiguous(), x_len)
        print(y[0].shape, y[1].shape)
        print("======================")
        e = edgemodel(S_ij[:, 0, :MAX_TIMESTEP].contiguous().to(device), y[1][:, 0, ].contiguous())
        print(e.shape)
        break
