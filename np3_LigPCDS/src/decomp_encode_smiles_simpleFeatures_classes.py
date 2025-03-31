from pathlib import Path
import sys
import pandas as pd
import rdkit.Chem as chem
from chemutils import sanitize, get_smiles, get_mol, restore_aromatics
from encode_ligs_xyz import Vocab
from mol_tree import MolTree


def smiles_class_frequencies(smiles_tree, vocab):
    freq = [0]*vocab.size()
    for node in smiles_tree:
        class_idx = vocab.get_index(node.smiles)
        class_count = round(len(node.clique)/vocab.get_atoms_count(class_idx))
        freq[class_idx] = freq[class_idx] + class_count
    return freq

def mol_class_frequencies(atoms_class, vocab):
    freq = [0]*vocab.size()
    for class_list in atoms_class.values():
        for c in class_list:
            freq[c] = freq[c] + 1
    freq = [x/vocab.get_atoms_count(j) for j, x in enumerate(freq)]
    return freq

# list the valid ligands in sdf files; sanitize and if valid store its informations
# if the list of valid ligands is supplied together with the vocab, then the script will skip to the encoding
# if only the list of valid ligands is supplied, then the script will skip to the vocab creation
# using the ligands smiles
def smiles_decompose_encode_to_cover_features_classes(ligands_file_path, output_path, features_simpler=False,
                                                      cycle_simpler=False, atoms_simpler=False, bonds_simpler=False,
                                                      smiles_col=None):
    # check inputs
    ligands_file_path = Path(ligands_file_path)
    if not ligands_file_path.exists() or not ligands_file_path.is_file():
        sys.exit("The provided ligand file does not exists.")
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()
        if not output_path.exists() or not output_path.is_dir():
            sys.exit("The provided output path does not exists and could not be created.")


    # read ligands csv file
    if smiles_col is None: # no header, ligands smiles are in the first column
        ligand_entries = pd.read_csv(ligands_file_path, na_values = ['null', 'N/A'],
                               keep_default_na = False, header=None)
        smiles_col = 0
    else:
        ligand_entries = pd.read_csv(ligands_file_path, na_values=['null', 'N/A'],
                                     keep_default_na=False)
        # check if column exists
        if not smiles_col in ligand_entries.columns:
            sys.exit("The provided smiles column is not present in the ligands_file_path.")
        smiles_col = ligand_entries.columns.get_loc(smiles_col)

    # ligand_entries = ligand_entries.iloc[1:1000, :]

    print("*** Start making the ligands vocabulary ***\n")
    # create a vocabulary from the features trees fragments of each unique smile and save it
    cset = set()
    i = 0
    smiles_class = {}
    for smiles in ligand_entries.iloc[:,smiles_col]:
        i = i + 1
        mol = MolTree(smiles, features_simpler=features_simpler, cycle_simpler=cycle_simpler,
                      atoms_simpler=atoms_simpler, bonds_simpler=bonds_simpler)
        smiles_class[smiles] = []
        for c in mol.nodes:
            cset.add(c.smiles)
            if 'H' in c.smiles:
                print('smiles ',smiles, ' returned fragment with H: ', c.smiles)
            smiles_class[smiles].append(c)
    cset = list(cset) # fix order

    print("\nVocabulary:\n")
    with open(output_path / str('vocabulary_smiles_simple_features_'+str(features_simpler)+'_cycles_'+str(cycle_simpler)+'_'
                  'atoms_'+str(atoms_simpler)+'_bonds_'+str(bonds_simpler)+'.txt'), 'w') as fo:
        for x in cset:
            print(x)
            fo.write(str(x) + "\n")

    print("\nVocabulary = "+str(len(cset))+
          " fragments \nLigand's unique SMILES = " + str(i) + "\n")
    vocab = Vocab(cset)
    del cset
    print("DONE!\n")


    print("*** Start extracting the ligand smiles class frequency ***\n")
    # add the class distribution for each ligand and
    # concat the ligand_entries_classes to the ligand_entries data frame
    ligand_entries_classes = pd.DataFrame([], columns=range(vocab.size()))
    i = 0
    for smiles in ligand_entries.iloc[:,smiles_col]:
        if i%1000 == 0:
            print("** Adding ligand smiles class frequency - entry "+str(i)+ " **\n")
        i = i + 1
        if not isinstance(smiles_class[smiles][0],int): # did not compute the class frequency yet
            smiles_class[smiles] = smiles_class_frequencies(smiles_class[smiles], vocab)
        ligand_entries_classes=ligand_entries_classes.append(pd.Series(smiles_class[smiles],
                          index=ligand_entries_classes.columns), ignore_index=True)

    # concat the ligand_entries_classes to the ligand entries data frame
    ligand_entries = pd.concat([ligand_entries.reset_index(drop=True),
                                ligand_entries_classes], axis=1)
    ligand_entries.to_csv(output_path / str('ligands_smiles_class_freq_simple_features_'+str(features_simpler)+'_cycles_'+str(cycle_simpler)+'_'
                  'atoms_'+str(atoms_simpler)+'_bonds_'+str(bonds_simpler)+'.csv'), index=False)
    print("DONE!")


if __name__ == "__main__":
    # read the ligands folder path
    if len(sys.argv) > 7:
        ligands_file_path = sys.argv[1]
        output_path = sys.argv[2]
        features_simpler = (sys.argv[3] == 'True')
        cycle_simpler = (sys.argv[4] == 'True')
        atoms_simpler = (sys.argv[5] == 'True')
        bonds_simpler = (sys.argv[6] == 'True')
        smiles_col = sys.argv[7]
    else:
        sys.exit("Wrong number of arguments. Seven arguments must be supplied in order to create a vocabulary "
                 "from a smiles list, encode the smiles with it and also store their classes distribution: \n"
                 "  1. ligands_file_path : path to the csv with the ligands smiles to be used;\n"
                 "  2. output_path : path to the folder where the output should be stored, if it does not exists create it;\n"
                 "  3. features_simple : bool True or False;\n"
                 "  4. cycle_bonds_simple : bool True or False;\n"
                 "  5. atoms_simple : bool True or False;\n"
                 "  6. bonds_simple : bool True or False;\n"
                 "  7. smiles_col : the name of the column in the ligands_file_path CSV where the ligands smiles are stored.\n"
                 )

    print('Simple features = ', features_simpler,'\nCycles = ', cycle_simpler, '\nAtoms = ', atoms_simpler,
          '\nBonds = ', bonds_simpler)
    smiles_decompose_encode_to_cover_features_classes(ligands_file_path, output_path, features_simpler,
                                                      cycle_simpler, atoms_simpler, bonds_simpler,
                                                      smiles_col)
