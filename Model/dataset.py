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
                # print(S_i[i, self.max_dim-i-1:])
                # print(A_exs[i, :i+1])
                S_i[i, self.max_dim-i-1:] = A_exs[i, :i+1]
            else:
                S_i[i, ] = A_exs[i, i-12:i]
        C = np.argmax(X, axis=1)
        atom_num = mol.GetNumAtoms()

        # create Input for EdgeRNN
        S_ij = np.zeros([self.seq_len, self.max_dim])
        for i in range(A.shape[0]):
            if i < self.max_dim:
                S_i[i, self.max_dim-i-1:] = A[i, :i+1]
            else:
                S_i[i, ] = A[i, i-12:i]

        # convert numpy.ndarray to torch.tensor
        S_i = torch.tensor(S_i, dtype=torch.float)
        S_ij = torch.tensor(S_ij, dtype=torch.long)
        C = torch.tensor(C, dtype=torch.long)

        return S_i, C, S_ij, atom_num


if __name__ == "__main__":
    smiles_list = read_smilesset("data/zinc_250k.smi")
    train_dataset = MolDataset(smiles_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    nodemodel = NodeRNN(input_size_adj=12, input_size_node=10, emb_size=64, hidden_lstm_size=128, num_layers=2,
                        hidden_header_size=64, out_features=10, seq_len=40)
    edgemodel = EdgeRNN(input_size=4, emb_size=128, hidden_lstm_size=128, num_layers=2, hidden_header_size=64,
                        out_features=4, seq_len=12)
    nodemodel.to(device)
    edgemodel.to(device)

    for i, (S_i, C, S_ij, x_len) in enumerate(train_loader):
        y = nodemodel(S_i.cuda(), C.cuda(), x_len)
        print(y[0].shape, y[1].shape)
        e = edgemodel(S_ij[:, 0, ].cuda(), y[1][:, 0])
        print(e.shape)
        break


