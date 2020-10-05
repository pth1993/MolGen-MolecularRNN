import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


if __name__ == "__main__":
    smiles = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
    A, X = Smiles2Graph(smiles)
    mol = orderBFSmol(A, X)
    DrawMol(mol, "data/result/structure/sample.png", size=(1000, 1000))

    # rwmol = Chem.RWMol()
    # rwmol.AddAtom(Chem.Atom(6))
    # print(rwmol.GetNumAtoms())
    # mol = rwmol.GetMol()
    #
    # DrawMol(Chem.MolToSmiles(mol), "data/result/structure/sample.png", size=(1000, 1000))

