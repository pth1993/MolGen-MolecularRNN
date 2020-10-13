import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


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
        S_i = np.array([self.seq_len, self.max_dim])
        for i in range(A.shape[0]):
            if i < self.max_dim:
                S_i[i, self.max_dim-i-1:] = A_exs[i, :i+1]
            else:
                S_i[i, ] = A_exs[i, i-12:i]
        C = np.argmax(X, axis=1).reshape([-1, 1])

        # create Input for EdgeRNN
        S_ij = np.array([self.seq_len, self.max_dim])
        for i in range(A.shape[0]):
            if i < self.max_dim:
                S_i[i, self.max_dim-i-1:] = A[i, :i+1]
            else:
                S_i[i, ] = A[i, i-12:i]

        # convert numpy.ndarray to torch.tensor
        S_i = torch.tensor(S_i, dtype=torch.long)
        S_ij = torch.tensor(S_ij, dtype=torch.long)
        C = torch.tensor(C, dtype=torch.long)

        return S_i, C, S_ij


