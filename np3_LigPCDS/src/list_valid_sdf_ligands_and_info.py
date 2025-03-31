from pathlib import Path
import sys
import pandas as pd
import rdkit.Chem as chem
from chemutils import sanitize, get_smiles, get_mol, restore_aromatics, add_CNOFH_charges
from encode_ligs_xyz import pdb_name
from Bio.PDB import PDBParser
from statistics import mean
import numpy as np

def np_mean_min_max_std(x):
    return round(np.mean(x), 4), round(np.min(x), 4), round(np.max(x), 4), round(np.std(x), 4)

def atom_mean_info_value(atom):
    num_disordered_units = atom.is_disordered()
    if num_disordered_units > 0:  # is disordered, get the first in list
        bf, occ = [], []
        for a in atom.disordered_get_list():
            bf.append(a.bfactor)
            occ.append(a.occupancy)
        bf = mean(bf)
        occ = mean(occ)
    else:
        bf = atom.bfactor
        occ = atom.occupancy
    return [occ, bf, num_disordered_units]

# list the valid ligands in sdf files present in the db; sanitize them and if valid store their information
def list_valid_sdf_ligands_and_info(db_ligand_path):
    # if vocab_path == '':
    #     vocab_path=None

    # check inputs
    # print(db_ligand_path)
    db_ligand_path = Path(db_ligand_path)
    if not db_ligand_path.exists() or not db_ligand_path.is_dir():
        sys.exit("The provided data folder do not exists.")

    # PDB parser to retrieve the ligands occupancy and bfactor by atom
    parser = PDBParser(PERMISSIVE=1)

    # store the valid ligands sdf present in the db path informations: b factor
    ligand_entries = []
    i = 0
    errors_count = []
    print("*** Start listing and validating ligands ***")
    for sdf in db_ligand_path.glob("*_NO_H.sdf"):
        #print(sdf.name)
        if i%100 == 0:
            print("** Processing file "+str(i)+ " **\n")
        i = i + 1
        # try to process the sdf to check if the ligand have valid definition
        try:
            mol_res = chem.SDMolSupplier(sdf.as_posix(), removeHs=True)
            mol_res = mol_res[0]
            # chem.Kekulize(mol_res)
            if add_CNOFH_charges(get_smiles(mol_res), kekule=True) is None:
                print('ERROR parsing sdf ' + sdf.name + "\n")
                errors_count.append(sdf.name)
                continue
            if mol_res.GetNumAtoms() == 0:
                print('ERROR no atoms in sdf ' + sdf.name + "\n")
                errors_count.append(sdf.name)
                continue
        except:
            print('ERROR loading ligand ' + sdf.name)
            errors_count.append(sdf.name)
            continue
        try:
            smiles_res = mol_res.GetProp("SMILES")
        except:
            smiles_res = get_smiles(mol_res, kekule=False) # leave empty
        # read the ligand .pdb file and retrieve the atoms occupancy and bfactor
        if not (db_ligand_path / pdb_name(sdf.name)).exists():
            print('ERROR the ligand pdb file '+ pdb_name(sdf.name)+ ' does not exists.')
            errors_count.append(sdf.name)
            continue 
        structure = parser.get_structure(sdf.name[0:-9], db_ligand_path / pdb_name(sdf.name))
        # get the atoms info in a df [occ, bf, num_disordered_units]
        atoms_info = pd.DataFrame.from_records([atom_mean_info_value(atom) for atom in structure.get_atoms()])
        occ = min(atoms_info[0])
        bf, bf_min, bf_max, bf_std = np_mean_min_max_std(atoms_info[1])
        num_disordered_units = sum(atoms_info[2])
        # store ligands info:
        # columns=['ligID', 'entry', 'bfactor', 'bfactor_min', 'bfactor_max', 'bfactor_std', 'min_occupancy', 'smiles', 'missingHeavyAtoms', 'numDisordered']
        ligand_entries.append([sdf.name.replace('_NO_H.sdf', ''),
                              sdf.name[5:8].rstrip('_'),
                              sdf.name[0:4], bf, bf_min, bf_max, bf_std, occ, smiles_res,
                              str(True).upper(),
                              #str(mol_res.GetPropsAsDict()['MissingHeavyAtoms']>0).upper(),  # this field is not present in the sdf file anymore
                              num_disordered_units])

    ligand_entries = pd.DataFrame(ligand_entries,
                 columns=['ligID', 'ligCode', 'entry', 'bfactor', 'bfactor_min', 'bfactor_max', 'bfactor_std',
                          'min_occupancy', 'smiles', 'missingHeavyAtoms', 'numDisordered'])
    # store valid ligands infos
    ligand_entries.to_csv(db_ligand_path.name+'_valid_sdf_info.csv', index=False)
    print("DONE!\n")
    if len(errors_count) > 0:
        print("* A total of "+str(len(errors_count))+"/"+str(i)+" ligands .sdf's raised an error and could not be processed *\n")
        print(errors_count)

if __name__ == "__main__":
    # read the ligands folder path
    if len(sys.argv) >= 2:
        db_ligand_path = sys.argv[1]
    else:
        sys.exit("Wrong number of arguments. One parameter must be supplied in order to create a list of the available "
                 "ligands, which were retrieved and have a valid sdf, and are present in the provided data folder. Their information is stored in the resulting list.\nParameter: \n"
                 "  1. ligands_data_folder: The path to the data folder where the SDF files of the retrieved ligands are located. "
                 "\nResult: One table will be created in the current directory named: "
                 "- ligands_data_folder.name+'_valid_sdf_info.csv': containing the list of available ligands with a valid SDF file and their information."
                 )
    list_valid_sdf_ligands_and_info(db_ligand_path)
