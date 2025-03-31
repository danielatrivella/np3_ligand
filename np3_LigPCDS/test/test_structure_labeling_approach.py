import sys

sys.path.insert(1, './src')
from chemutils import label_mol_atoms, get_mol
import pandas as pd
import numpy as np

def check_structure_labeling(ligands_test, SP_base=True):
    n = ligands_test.shape[0]
    ligs_error = {}

    for i, smiles in enumerate(ligands_test.smiles):
        print("Testing smiles", smiles, "(",i+1,"/",n,")")
        mol = get_mol(smiles, addHs=True)
        atoms_sp_class = np.asarray(label_mol_atoms(mol, steric_number=SP_base))
        # check classes
        if SP_base:
            expected_classes = np.asarray(ligands_test.Labels[i].split(';'))
        else:
            expected_classes = np.asarray([atom_class.split(":")[0] for atom_class in ligands_test.Atoms[i].split(';')])
        if not np.all(expected_classes == atoms_sp_class):
            wrong_labels = np.where(expected_classes != atoms_sp_class)
            ligs_error[ligands_test.lig_code[i]] = [wrong_labels,
                                                 atoms_sp_class[wrong_labels],
                                                 expected_classes[wrong_labels]]

    if len(ligs_error) > 0:
        print("ERROR in", len(ligs_error), "ligands")
        for lig in ligs_error:
            print("-",lig)
            print("  Wrong labeled atoms:", ligs_error[lig][0])
            print("  Given labels:", ligs_error[lig][1])
            print("  Expected labels:", ligs_error[lig][2])
    else:
        print("\nNo errors were found. :)\nDone!")

ligands_test = pd.read_csv("test/ligands_label/ligands_label.csv", na_values = ['null', 'N/A'],
                           keep_default_na = False)

# SP-base testing
print("\n** Testing the structure labeling for SP-based modeling **\n")
check_structure_labeling(ligands_test, SP_base=True)
# Atom-base testing
print("\n\n** Testing the structure labeling for Atom-based modeling **\n")
check_structure_labeling(ligands_test, SP_base=False)
