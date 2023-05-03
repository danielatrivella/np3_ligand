import rdkit.Chem as chem
from pathlib import Path
import sys
from shutil import rmtree
from chemutils import label_mol_atoms, restore_aromatics, get_mol, get_smiles, get_clique_mol
import pandas as pd
from statistics import mean
from Bio.PDB import PDBParser

# convert the smiles class list to a frequency vector
def smiles_SP_class_frequencies(SP_class_list, vocab):
    freq = [0]*len(vocab)
    class_idxs = []
    for SP_class in SP_class_list:
        class_idx = vocab.index(SP_class)
        freq[class_idx] = freq[class_idx] + 1
        class_idxs.append(class_idx)
    return class_idxs, freq

def mean_list(list):
    if len(list) == 0:
        return "N/A"
    else:
        return round(mean(list),4)

def format_coord(coord):
    return str(int(coord[0]))+str(int(coord[1]))+str(int(coord[2]))

def atom_info_key(atom):
    if atom.is_disordered() > 0:  # is disordered, get the first in list
        atom = atom.disordered_get_list()[0]
    return str(atom.element.upper() + format_coord(atom.coord))

def atom_info_value(atom):
    num_disordered_units = atom.is_disordered()
    if num_disordered_units > 0:  # is disordered, get the first in list
        atom = atom.disordered_get_list()[0]
    return [atom.occupancy, atom.bfactor, num_disordered_units]

# from a sdf file name to a pdb name
def pdb_name(file_name, pdb_suffix='_lig.pdb'):
    file_name = file_name.split('_')
    return str(file_name[0]+'_'+file_name[1]+'_'+file_name[3]+'_'+file_name[2]+pdb_suffix)

def bounding_box(set_points):
    min_p = min(set_points)
    max_p = max(set_points)
    center_point = round(mean([min_p, max_p]), 4)
    box_edge = round(max_p-min_p, 4) + 2.1*2 # add a gap equals the diameter of the biggest atom
    # return max dist
    return [center_point, box_edge]

def match_substructure(mol_ref, mol):
    map_atoms = mol_ref.GetSubstructMatch(mol)
    if len(map_atoms) == 0: # try to map inverse
        map_atoms = mol.GetSubstructMatch(mol_ref)
        if len(map_atoms) > 0: # invert the mapping
            map_atoms = [map_atoms.index(i) for i in range(len(map_atoms))]
    return map_atoms

def encode_ligands_vocabulary(db_lig_path, valid_ligs_file, vocab_path, label_SN, n, i_start):
    if n <= i_start and n > 0:
        sys.exit("The provided stop row is not greater than the start row. Wrong range.")

    # check inputs
    db_lig_path = Path(db_lig_path)
    if not db_lig_path.exists() or not db_lig_path.is_dir():
        sys.exit("The provided data folder do not exists.")
    if not Path(valid_ligs_file).is_file():
        sys.exit("The provided valid Ligands list CSV file do not exists.")
    if not Path(vocab_path).is_file():
        sys.exit("The provided vocabulary text file do not exists.")

    # create the directory xyz if it does not exists yet
    xyz_dir_name = str('xyz_'+Path(valid_ligs_file).name.replace('.csv', ''))
    try:
        Path(db_lig_path / xyz_dir_name).mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        while True:
            print("The output directory "+Path(db_lig_path / xyz_dir_name).as_posix()+" already exists. \n "
                  "Do you want to remove it? (y - yes|n - no|o - overwrite)")
            rm_dir = input()
            if rm_dir in ["y", "yes", "n", "no", "o", "overwrite"]:
                break
        if rm_dir in ["y", "yes"]:
            rmtree(Path(db_lig_path / xyz_dir_name))
            Path(db_lig_path / xyz_dir_name).mkdir(parents=True, exist_ok=False)
        elif rm_dir in ["o", "overwrite"]:
            print("The directory content will be overwritten.")
        else:
            print("Change the existing directory name and retry.")
            sys.exit(1)

    # read csv with the valid ligands list to encode
    ligs_retrieve = pd.read_csv(valid_ligs_file, na_values = ['null', 'N/A'], keep_default_na = False) # do not interpret sodium NA as nan
    if n == 0 or n > ligs_retrieve.shape[0]:
        n = ligs_retrieve.shape[0]

    # read the vocabulary and store it in a list (ordered)
    vocab = [line.rstrip('\n') for line in open(vocab_path)]

    # store the retrieved ligands informations: bounding box, class freq and # H's
    ligand_label = [] #pd.DataFrame([], columns=['entryID', 'x', 'y', 'z','x_bound','y_bound','z_bound', 'occupancy', 'bfactor','MissingHeavyAtoms', 'class_freq'])

    # PDB parser to retrieve the ligands occupancy and bfactor by atom
    parser = PDBParser(PERMISSIVE=1)
    #
    # for each structure select all sdf's from the available ligands and
    # encode each ligand using the provided vocabulary
    sdf_errors = []
    # sort by smiles to put the empty smiles in the end and by ligCode to prevent recomputing ligand's smiles classes
    ligs_retrieve = ligs_retrieve.sort_values(["ligCode","missingHeavyAtoms"]).reset_index(drop=True)
    lig_code = "" # store the last lig code of the corresponding smiles for which the SP classes where computed
    lig_smiles_SP_class = [] # store the last SP classes of the corresponding smiles
    ligs_retrieve["match_structure"] = True
    ligs_retrieve["filter_quality"] = True
    for i in range(i_start,n):
        # print(i)
        if i%100 == 0:
            print("\n** Processing ligand entry "+str(i)+ "/"+str(n)+" **\n")
        sdf = db_lig_path / str(ligs_retrieve.ligID[i]+"_NO_H.sdf")
        #
        # try to process the sdf to check if the ligand have valid definition
        try:
            mol_res = chem.SDMolSupplier(sdf.as_posix(), removeHs=True)
            mol_res = mol_res[0]
            if mol_res is None:
                print('ERROR parsing sdf ' + sdf.name + "\n")
                sdf_errors.append(sdf.name)
                continue
            if mol_res.GetNumAtoms() == 0:
                print('ERROR no atoms in sdf ' + sdf.name + "\n")
                sdf_errors.append(sdf.name)
                continue
            chem.Kekulize(mol_res)
        except:
            print('ERROR loading ligand ' + sdf.name)
            sdf_errors.append(sdf.name)
            continue
        #
        try:
            mol = chem.AddHs(mol_res)
            # retrieve the SP class of each atom, remove the hydrogens (last labels)
            atoms_sp_class = label_mol_atoms(mol, steric_number=label_SN)
            #
            # test if the sp classes match the corresponding smiles sp classes
            # retrieve the sp classes of the corresponding smiles
            if lig_code != ligs_retrieve.ligCode[i] or ligs_retrieve.loc[i,"missingHeavyAtoms"]:
                # smiles present in the sdf file
                lig_code = ligs_retrieve.ligCode[i]
                mol_ref = get_mol(ligs_retrieve.smiles[i], addHs=True)
                if not mol_ref:
                    print("\n***\nERROR with ligand smiles idx " + str(i) +
                          " from the sdf " + sdf.as_posix() + " the smiles is not correctly parsed into a mol: " +
                          ligs_retrieve.smiles[i]+"\n***\n")
                    mol_ref = get_mol(get_smiles(mol_res), addHs=True)
                lig_smiles_SP_class = label_mol_atoms(mol_ref, steric_number=label_SN, notH=False)
                # restore mol ref to compare with the sdf mol
                mol_ref = chem.RemoveHs(mol_ref)
                chem.Kekulize(mol_ref)
                restore_aromatics(mol_ref)
            # match reference mol with the sdf mol
            mol = chem.RemoveHs(mol)
            restore_aromatics(mol)
            map_atoms = match_substructure(mol_ref, mol)
            # check if classes are equal
            if len(map_atoms) == len(atoms_sp_class): # mapping was successful
                if not atoms_sp_class == [lig_smiles_SP_class[j] for j in map_atoms]:
                    # if missing heavy atoms also compare the classes using the matching substructure instead of only the original reference
                    if ligs_retrieve.missingHeavyAtoms[i]:
                        chem.Kekulize(mol_ref)
                        mol_ref_sub = get_clique_mol(mol_ref, list(map_atoms))
                        restore_aromatics(mol_ref)
                        restore_aromatics(mol_ref_sub)
                        map_atoms = match_substructure(mol_ref_sub, mol)
                        lig_sub_smiles_SP_class = label_mol_atoms(chem.AddHs(mol_ref_sub), steric_number=label_SN,
                                                                     notH=False)
                        if not atoms_sp_class == [lig_sub_smiles_SP_class[j] for j in map_atoms]:
                            ligs_retrieve.loc[i, "filter_quality"] = False
                    else:
                        ligs_retrieve.loc[i, "filter_quality"] = False
            else: # mapping failed when matching structures
                ligs_retrieve.loc[i, "match_structure"] = False
                # remove hydrogen class before matching
                lig_smiles_SP_class = [x for x in lig_smiles_SP_class if not x == '-1']
                # if sdf is complete, then the smiles classes must be equal to the sdf classes
                if not ligs_retrieve.missingHeavyAtoms[i] and not sorted(atoms_sp_class) == sorted(lig_smiles_SP_class):
                    ligs_retrieve.loc[i, "filter_quality"] = False
                # if sdf is missing atoms, the sdf classes will be a subset of the smiles classes
                elif ligs_retrieve.missingHeavyAtoms[i] and not set(atoms_sp_class).issubset(lig_smiles_SP_class):
                    # if missing atoms and not a subset, then the complete smiles is different from the fragment - possible aromatic atoms are different
                    # try to match structure after kekulize - remore aromaticity info
                    chem.Kekulize(mol_ref)
                    chem.Kekulize(mol)
                    map_atoms = match_substructure(mol_ref, mol)
                    if len(map_atoms) == len(atoms_sp_class): # mapping after kekulize was successful
                        # if the matched classes are still not equal, then get a substructure from the smiles that matches the mol sdf and recompute the reference classes from the smiles substructure to check again if they match
                        if not atoms_sp_class == [lig_smiles_SP_class[j] for j in map_atoms]:
                            # if classes are different, then get the matched substructed from the smile mol_ref and recompute the sub mol_ref labels                        
                            mol_ref_sub = get_clique_mol(mol_ref, list(map_atoms))
                            restore_aromatics(mol_ref)
                            restore_aromatics(mol_ref_sub)
                            restore_aromatics(mol)
                            # match mol_ref substructure with the sdf mol
                            map_atoms = match_substructure(mol_ref_sub, mol)
                            # relabel the mol_ref substructure and check labels equality to the sdf mol
                            lig_sub_smiles_SP_class = label_mol_atoms(chem.AddHs(mol_ref_sub), steric_number=label_SN,
                                                                         notH=False)
                            # if the classes are still not equal, than remove lig entry
                            if not atoms_sp_class == [lig_sub_smiles_SP_class[j] for j in map_atoms]:
                                ligs_retrieve.loc[i, "filter_quality"] = False   
                    else:                     
                        ligs_retrieve.loc[i, "filter_quality"] = False
        except Exception as e:
            print("***\nError creating mol, encoding and matching with corresponding smiles. SDF ", sdf.as_posix(),"\n***",
                  e, e.with_traceback())
            sys.exit(1)
        #
        #
        if ligs_retrieve.loc[i, "filter_quality"]:
            atoms_sp_class_idx, atoms_sp_class_freq = smiles_SP_class_frequencies(atoms_sp_class, vocab)
            #
            # read the ligand .pdb file and retrieve the atoms occupancy and bfactor
            # in a dictionary: keys are atom.element+format_coord(atoms.coords)
            if (db_lig_path/pdb_name(sdf.name)).exists():
                structure = parser.get_structure(sdf.name[0:-9], db_lig_path/pdb_name(sdf.name))
                atoms_info = {atom_info_key(atom): atom_info_value(atom) for atom in structure.get_atoms()}
            else:
                atoms_info = None
            #
            # store the ligand atoms coordinates, occupancy and bfactor
            x, y, z = [], [], []
            occ, bf = [], []
            # write the ligand xyz file with the atoms classes in this vocab
            # print("Writting "+ str(db_lig_path / xyz_dir_name / sdf.name.replace('NO_H.sdf', 'class.xyz')))
            with open(db_lig_path / xyz_dir_name / sdf.name.replace('NO_H.sdf', 'class.xyz'), 'w') as fo:
                fo.write('index, symbol, x, y, z, occupancy, bfactor, numDisordered, labels\n') #occupancy, bfactor, numDisordered, implicitValence, labels\n')
                for j, atom in enumerate(mol_res.GetAtoms()):
                    # print(i, atom.GetSymbol().upper())
                    if atom.GetSymbol().upper() in ["H","D"]: # skip dummy atoms
                        continue
                    atom_class = atoms_sp_class_idx[j]
                    x.append(mol_res.GetConformer().GetAtomPosition(j).x)
                    y.append(mol_res.GetConformer().GetAtomPosition(j).y)
                    z.append(mol_res.GetConformer().GetAtomPosition(j).z)
                    if atoms_info:
                        try:
                            atom_occ_bf = atoms_info[str(atom.GetSymbol().upper() +
                                                     format_coord([x[-1], y[-1], z[-1]]))]
                            occ.append(atom_occ_bf[0])
                            bf.append(atom_occ_bf[1])
                            atom_occ_bf = str(atom_occ_bf).strip('[]')
                        except KeyError:
                            print("\n***\nERROR matching atom "+ str(i)+
                                  " from the sdf "+sdf.as_posix()+" to the pdb using its coords and symbol: "+
                                  str(atom.GetSymbol().upper()+format_coord([x[-1], y[-1], z[-1]]))+
                                  "\n***\n")
                            atom_occ_bf = 'N/A,N/A,N/A'
                    else:
                        atom_occ_bf = 'N/A,N/A,N/A'
                    # write ligand atoms and labels
                    fo.write(str(atom.GetIdx()) + ',' +
                             atom.GetSymbol() + ',' +
                             str(x[-1]) + ',' + str(y[-1]) + ',' + str(z[-1]) + ',' + atom_occ_bf + ',' +
                             #str(mol_res.GetAtoms()[i].GetImplicitValence()) + ',' +
                             '"' + str(atom_class) + '"\n')
            # compute the ligand middle point (the mean of the extrema),
            # search radius (max distance to an atom) and classes frequency for the valid ligands
            ligand_box = bounding_box(x), bounding_box(y), bounding_box(z)
            ligand_label.append([sdf.name.replace('_NO_H.sdf', ''),
                                 ligand_box[0][0], ligand_box[1][0], ligand_box[2][0],
                                 ligand_box[0][1], ligand_box[1][1], ligand_box[2][1]] +
                                atoms_sp_class_freq)
        else:
            ligand_label.append([sdf.name.replace('_NO_H.sdf', ''),
                                0, 0, 0,
                                 0, 0, 0] +
                                [0]*len(vocab))

    # concat the encoded ligands info
    ligand_label = pd.DataFrame(ligand_label,
                                columns=['ligID', 'x', 'y', 'z', 'x_bound', 'y_bound', 'z_bound']+list(range(len(vocab))))
    # filter only the encoded ligands
    ligs_retrieve = ligs_retrieve.merge(ligand_label, on='ligID')
    ligs_retrieve.to_csv(db_lig_path / xyz_dir_name / Path(valid_ligs_file).name.replace('.csv','_box_class_freq.csv'),
                        index=False)
    #
    if len(sdf_errors) > 0:
        print("A total of "+str(len(sdf_errors))+" ligands sdf raised an error:\n"+str(sdf_errors))
    print("* A total of " + str(
        len(ligs_retrieve.entry.unique()) - len(ligs_retrieve.entry[ligs_retrieve.filter_quality].unique())) +
          "/" + str(len(ligs_retrieve.entry.unique())) +
          " PDB entries were removed due to the quality filter *\n")
    print("* A total of " + str(ligs_retrieve.shape[0] - ligs_retrieve.filter_quality.sum()) + "/" + str(ligs_retrieve.shape[0]) +
          " ligands were removed due to the quality filter - not matching SP classes *\n")


if __name__ == "__main__":
    # read the ligands csv with respective pdb entries
    i_start = n = 0
    label_SN = True
    if len(sys.argv) >= 4:
        db_lig_path = sys.argv[1]
        valid_ligs_file = sys.argv[2]
        vocab_path = sys.argv[3]
        if len(sys.argv) > 4:
             label_SN = (sys.argv[4].lower() == "true")
        if len(sys.argv) > 5:
            i_start = int(sys.argv[5])
        if len(sys.argv) > 6:
            n = int(sys.argv[6])
    else:
        sys.exit("Wrong number of arguments. Three argument must be supplied in order to read the ligands "
                 ".sdf data and a vocabulary text file, and then label the ligands atoms using it: \n"
                 "  1. The path to the data folder called 'ligands' where the ligands sdf files are located. "
                 "One folder will be created inside it: 'xyz_<ligand csv name>' to store the coordinates files of the "
                 "ligand's atoms labeled with the given vocabulary;\n"
                 "  2. The path to the CSV file containing the valid ligands list and their IDs. "
                 "This file is expected to be the output of the quality filter script."
                 "Mandatory column = 'ligID'.;\n"
                 "  3. The path to the text file containing the desired vocabulary to be used to label the ligands. "
                 "It must contain one class per line. The ligands SDF will be fragmented and matched against this "
                 "list to be labeled using the vocabulary index order;\n"
                 "  4. label_SP: (optional) True to use the SP classes in the vocabulary creation (default), otherwise use the atom symbol. "
                 "Both cases will also add the cycles information for each atom.\n"
                 "  5. (optional) The number of the row of the ligands CSV file where the script should start. "
                 "Skip to the given row or, if missing, start from the beginning;\n"
                 "  6. (optional) The number of the row of the ligands CSV file where the script should stop. "
                 "Stop in the given row or, if missing, stop in the last row.")
    encode_ligands_vocabulary(db_lig_path, valid_ligs_file, vocab_path, label_SN, n, i_start)
