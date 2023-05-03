import sys
from pathlib import Path
import pandas as pd
from chemutils import label_mol_atoms, get_mol

# def unique_smiles(smiles_list):
#     if len(smiles_list) == 1:
#         return smiles_list
#     unique_smiles = [smiles_list[0]]
#     for i in range(1,len(smiles_list)):
#         if smiles_list[i] not in unique_smiles:
#             unique_smiles.append(smiles_list[i])
#     return unique_smiles
#
# def extend_unique(full_list, new_list):
#     full_list.extend([elem for elem in new_list if elem not in full_list])

def create_vocabulary_ligand_smiles(output_path, valid_ligs_file, label_SN):
    db_path = Path(output_path)
    if not db_path.exists() or not db_path.is_dir():
        sys.exit("The provided output folder do not exists.")
    if not Path(valid_ligs_file).is_file():
        sys.exit("The provided Ligands list CSV file do not exists.")

    # read csv with the ligands smiles to use to create a vocab
    valid_ligs = pd.read_csv(valid_ligs_file, na_values=['null', 'N/A'],
                               keep_default_na=False)  # do not interpret sodium NA as nan

    valid_ligs_file = Path(valid_ligs_file).name.replace(".csv", "_"+("SPclasses" if label_SN else "atomSymbol"))
    # make sure there is no repeated smiles in the list
    ligands_smiles = valid_ligs.smiles.unique()
    n_entries = len(ligands_smiles)
    # save the unique smiles list from the sdf ligands
    with open(db_path/str('ligs_smiles_'+valid_ligs_file+'.txt'), 'w') as fo:
        for smiles in ligands_smiles:
            fo.write(str(smiles)+"\n")

    # create a vocabulary from each unique smile and save it
    cset = set()
    for smiles in ligands_smiles:
        mol = get_mol(smiles, addHs=True)
        if not mol:
            continue
        atoms_sp_class = label_mol_atoms(mol, steric_number=label_SN)
        # add each sp classes to the vocab
        for c in atoms_sp_class:
            cset.add(c)

    cset = list(cset)  # fix order
    print("\nVocabulary:\n")
    with open(db_path/str('vocabulary_'+valid_ligs_file+'.txt'), 'w') as fo:
        for x in cset:
            print(x)
            fo.write(str(x)+"\n")

    print("\nVocabulary = "+str(len(cset)) + " " + ("SP classes" if label_SN else "Atom Symbol classes") +
          " \nLigand's unique SMILES = " + str(n_entries) + "\n")


# def create_vocabulary_ligand_sdf(db_path, pdb_ligs_file, n, i_start=0):
#     if n <= i_start and n > 0:
#         sys.exit("The provided stop row is not greater than the start row. Wrong range.")
#
#     db_path = Path(db_path)
#     if not db_path.exists() or not db_path.is_dir():
#         sys.exit("The provided data folder do not exists.")
#     if not Path(pdb_ligs_file).is_file():
#         sys.exit("The provided PDB list CSV file do not exists.")
#
#     # read csv with the pdb list and ligands to use to create a vocab
#     pdb_retrieve = pd.read_csv(pdb_ligs_file, na_values=['null', 'N/A'],
#                                keep_default_na=False)  # do not interpret sodium NA as nan
#     if n == 0 or n > pdb_retrieve.shape[0]:
#         n = pdb_retrieve.shape[0]
#
#     sdf_smiles = [] # the smiles database to be retrieved from the available sdfs
#     n_entries = 0
#     # for each structure ID filter all sdf's from the available ligands and
#     # add each ligand to the vocabulary
#     for i in range(i_start, n):
#         pdb_id = pdb_retrieve.PDBID[i].lower()
#         print("\n** Structure " + pdb_id + "("+str(i+1)+"/"+str(n)+") **\n")
#
#         ligand_sdfs_smiles = []
#         # for each ligand present in the pdb id, try to locate its .sdf and
#         # add its smile to the vocabulary
#         sdf_files = np.array(list(db_path.glob(str("ligands/" + pdb_id + "_*.sdf"))))
#         for lig in pdb_retrieve.ligandsID[i].split(' '):
#             # sdf_files = db_path.glob(str("ligands/"+pdb_id+"_"+lig +"_*.sdf"))
#             sdf_lig_idx = [i for i in range(len(sdf_files)) if lig in sdf_files[i].name]
#             for i in sdf_lig_idx:
#                 sdf = sdf_files[i]
#             # for sdf in sdf_files:
#                 #print(sdf.as_posix())
#                 try:
#                     mol_res = chem.SDMolSupplier(sdf.as_posix(), removeHs=True)
#                     if sanitize(mol_res[0]) is None:
#                         print('ERROR parsing sdf ' + sdf.name + "\n")
#                         continue
#                     if mol_res[0].GetNumAtoms() == 0:
#                         print('ERROR no atoms in sdf ' + sdf.name + "\n")
#                         continue
#                     # elif mol_res[0].GetPropsAsDict()['MissingHeavyAtoms'] > 0: # check if the sdf is complete
#                     #         print('\nWarning MissingHeavyAtoms. Skipping sdf ' + sdf.name + "\n")
#                     #         continue
#                     print(sdf.name)
#                     mol_res = mol_res[0]
#                     chem.Kekulize(mol_res)
#                     ligand_sdfs_smiles.append(get_smiles(mol_res))
#                 except:
#                     print('ERROR loading ligand ' + sdf.name)
#                     continue
#             # remove the already processed sdf files
#             sdf_files = np.delete(sdf_files, sdf_lig_idx, 0)
#         # append the sdf smile to the database list if it was loaded correctly
#         if len(ligand_sdfs_smiles) == 0:
#             print('No sdf file available for this entry. Skipping...\n')
#             continue
#         # elif len(ligand_sdfs_smiles) > 1:
#         #         print("Filtering unique smiles\n")
#         #         ligand_sdfs_smiles = unique_smiles(ligand_sdfs_smiles)
#             #n_entries = n_entries + len(ligand_sdfs_smiles)
#         # print(str(ligand_sdfs_smiles))
#         # add the unique smiles list by ligand to the final smiles list
#         extend_unique(sdf_smiles,ligand_sdfs_smiles)
#
#     if len(sdf_smiles) == 0:
#         sys.exit("Error no ligand could be fragmented correctly.")
#     # add suffix naming the ligand csv table
#     pdb_ligs_file = Path(pdb_ligs_file).name.replace('.csv', '')
#
#     # make sure there is no repeated smiles in the list
#     sdf_smiles = unique_smiles(sdf_smiles)
#     n_entries = len(sdf_smiles)
#     # save the unique smiles list from the sdf ligands
#     with open(db_path/str('ligs_smiles_'+pdb_ligs_file+'.txt'), 'w') as fo:
#         for smiles in sdf_smiles:
#             fo.write(str(smiles)+"\n")
#
#     # create a vocabulary from the features trees fragments of each unique smile and save it
#     cset = set()
#     for smiles in sdf_smiles:
#         mol = MolTree(smiles, True)
#         for c in mol.nodes:
#             cset.add(c.smiles)
#
#     print("\nVocabulary:\n")
#     with open(db_path/str('vocabulary_'+pdb_ligs_file+'.txt'), 'w') as fo:
#         for x in cset:
#             print(x)
#             fo.write(str(x)+"\n")
#
#     print("\nVocabulary = "+str(len(cset))+" fragments \nLigand's unique SMILES = " + str(n_entries) + "\n")


if __name__ == "__main__":
    # lg = rdkit.RDLogger.logger()
    # lg.setLevel(rdkit.RDLogger.CRITICAL)
    # i_start = 0
    # n = 0
    label_SN=True
    # read the ligands csv with respective pdb entries
    if len(sys.argv) >= 3:
        output_path = sys.argv[1]
        ligs_file = sys.argv[2]
        if len(sys.argv) > 3:
             label_SN = (sys.argv[3].lower() == "true")
        # if len(sys.argv) > 4:
        #     n = int(sys.argv[4])
    else:
        sys.exit("Wrong number of arguments. Two arguments must be supplied in order to read the valid ligands "
                 "list, retrieve their smiles and create a vocabulary of their atoms. "
                 "A third parameter is supplied to choose between the SP classes labels and the atom symbol label: \n"
                 "  1. output_path: The path to the data folder where the vocabulary output will be stored"
                 "Two files will be created: "
                 "- 'ligs_smiles_<ligs_file.name>.txt' containing all the smiles used in the vocabulary creation (the data base) and "
                 "- 'vocabulary_<ligs_file.name>.txt' containing all SP classes resulting from the SP hybridization classification of the smiles's atoms (the vocabulary itself);\n"
                 "  2. ligs_file: The path to the CSV file containing the valid ligands list and their smiles. This file is expected to be the output of the quality filter script."
                 "Mandatory column = 'smiles'.\n"
                 "  3. SP label: (optional) True to use the SP classes in the vocabulary creation (default), otherwise use the atom symbol. "
                 "Both cases will also add the cycles information for each atom.\n")
    create_vocabulary_ligand_smiles(output_path, ligs_file, label_SN)
    # create_vocabulary_ligand_sdf(output_path, ligs_file)