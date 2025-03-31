import sys
import pandas as pd
from pathlib import Path
from rdkit import Chem
from mordred import Calculator, RingCount, AtomCount, BondCount
import numpy as np
from chemutils import get_mol, fix_valence_charge

if len(sys.argv) < 2:
    print(" Incorrect number of arguments")
    print(" 1. ligand smiles file path; 2. smiles column;")
    sys.exit()

ligand_smiles_path = Path(sys.argv[1])
smiles_col = sys.argv[2]

ligand_entries = pd.read_csv(ligand_smiles_path, na_values=['null', 'N/A'],
                                     keep_default_na=False)

# generate mols from smiles
mols = []
error_parsing = []
for smiles in ligand_entries.loc[:,smiles_col]:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        mol = get_mol(smiles, kekule=False)
        if mol is None:
            error_parsing.append(2)
        else:
            error_parsing.append(1)
    else:
        error_parsing.append(0)
    mols.append(mol)

# get the errors index that could not be fixed
null_mols = np.where([x == None for x in mols])[0]
# print(ligands_df.loc[null_mols])

# remove null Mols
mols = np.delete(mols, null_mols)
# extract ring and atoms descriptors from smiles
calc = Calculator()
calc.register(RingCount.RingCount) # ring info
calc.register(AtomCount.AtomCount) # atoms info
calc.register(BondCount.BondCount) # bonds info

mols_calc = calc.pandas(mols)
# add null entries
for x in null_mols:
    mols_calc = pd.DataFrame(np.insert(mols_calc.values, x, values=[-1]*len(mols_calc.columns), axis=0), columns=mols_calc.columns)

# concate result with the ligand table
ligand_entries = pd.concat([ligand_entries, mols_calc], axis=1)
ligand_entries = pd.concat([ligand_entries, pd.Series(error_parsing, name= 'Error_mol')], axis=1)


#output_path="cmb_ligands_descriptions.csv"
ligand_entries.to_csv(ligand_smiles_path.parent / ligand_smiles_path.name.replace('.csv', '_info_mordred.csv'),
                      index=False)
