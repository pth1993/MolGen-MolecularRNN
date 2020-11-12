import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
import random


if __name__ == "__main__":
    smiles = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
    # smiles = "c1ccccc1"
    # smiles = "CCC"

    # mol = Chem.MolFromSmiles(smiles)
    # A, X = Smiles2Graph(mol, is_Tensor=False)
    # mol = orderBFSmol(A, X, num_atom=mol.GetNumAtoms())
    # A, X = Smiles2Graph(mol, is_Tensor=False)
    # print(X.shape)
    # DrawMol(mol, "data/result/structure/sample_t.png", size=(1000, 1000))

    # rwmol = Chem.RWMol()
    # rwmol.AddAtom(Chem.Atom(6))
    # rwmol.AddAtom(Chem.Atom(6))
    # rwmol.AddBond(0, 1, Chem.BondType.SINGLE)
    # rwmol.AddAtom(Chem.Atom(6))
    # rwmol.AddBond(1, 2, Chem.BondType.SINGLE)
    # rwmol.AddAtom(Chem.Atom(6))
    # rwmol.AddBond(2, 3, Chem.BondType.SINGLE)
    # rwmol.AddAtom(Chem.Atom(6))
    # rwmol.AddBond(3, 4, Chem.BondType.SINGLE)
    # rwmol.AddBond(4, 0, Chem.BondType.SINGLE)
    # mol = rwmol.GetMol()
    # DrawMol(mol, "data/result/structure/sample.png", size=(1000, 1000))

    smiles_list = read_smilesset("data/zinc_250k_test.smi")
    random.shuffle(smiles_list)
    atoms = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        atoms.append(mol.GetNumAtoms())

    df = pd.DataFrame(columns=["SMILES", "Atoms"])
    df["SMILES"] = smiles_list
    df["Atoms"] = atoms
    df = df.sort_values("Atoms")

    smiles_list = df["SMILES"].to_list()
    with open("data/zinc_250k_test_sorted.smi", mode="w") as f:
        for smiles in smiles_list:
            f.write(smiles+"\n")


