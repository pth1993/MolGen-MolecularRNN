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

        S_i = np.zeros([self.seq_len+1, MAX_TIMESTEP])
        S_ij = np.zeros([self.seq_len, MAX_TIMESTEP+1])
        S_ij[:, 0] = np.ones(self.seq_len) * BOND_IDX["&"]
        for i in range(A.shape[0]):
            for j in range(min(MAX_TIMESTEP, i)):
                S_i[i+1, j] = 1 if A[i, i - 1 - j] > 0 else 0
                S_ij[i, j + 1] = A[i, i - 1 - j]

        C = np.argmax(X, axis=1)
        C = np.insert(C, 0, ATOM_IDX["&"])
        atom_num = mol.GetNumAtoms()

        # convert numpy.ndarray to torch.tensor
        S_i = torch.tensor(S_i, dtype=torch.float)
        S_ij = torch.tensor(S_ij, dtype=torch.long)
        C = torch.tensor(C, dtype=torch.long)

        return S_i, C, S_ij, atom_num
