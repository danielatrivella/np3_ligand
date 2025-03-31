from pathlib import Path
import sys
import pandas as pd
from chemutils import label_mol_atoms_SN, get_mol
from encode_ligs_xyz import smiles_SP_class_frequencies

# decompose the ligands smiles of the ligands_file_path data present in the smiles_col column using the SP classes to
# construct a vocabulary.
# Encode the smiles with the obtained vocabulary and compute the classes frequency
def smiles_decompose_encode_to_cover_SP_classes(ligands_file_path, output_path, smiles_col=None):
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

    # ligand_entries = ligand_entries.iloc[1:100, :]

    print("*** Start making the ligands vocabulary ***\n")
    # create a vocabulary from the sp classes list of each unique smile and save it
    cset = set()
    i = 0
    sp_class = {}
    for smiles in ligand_entries.iloc[:,smiles_col].unique():
        i = i + 1
        mol = get_mol(smiles, addHs=True)
        atoms_sp_class = label_mol_atoms_SN(mol)
        # add each sp classes to the vocab
        for c in atoms_sp_class:
            cset.add(c)

    cset = list(cset) # fix order

    print("\nVocabulary:\n")
    with open(output_path / str('vocabulary_SP.txt'), 'w') as fo:
        for x in cset:
            print(x)
            fo.write(str(x) + "\n")

    print("\nVocabulary = "+str(len(cset))+
          " classes \nLigand's unique SMILES = " + str(i) + "\n")
    print("DONE!\n")


    print("*** Start extracting the ligand smiles class frequency ***\n")
    # add the class distribution for each ligand and
    # concat the ligand_entries_classes to the ligand_entries data frame
    ligand_entries_classes = []
    i = 0
    for smiles in ligand_entries.iloc[:,smiles_col]:
        if i%1000 == 0:
            print("** Adding ligand smiles class frequency - entry "+str(i)+ " **\n")
        i = i + 1
        mol = get_mol(smiles, addHs=True)
        atoms_sp_class = label_mol_atoms_SN(mol)
        sp_class_idx, sp_class_freq = smiles_SP_class_frequencies(atoms_sp_class, cset)
        ligand_entries_classes.append(sp_class_freq)

    # concat the ligand_entries_classes to the ligand entries data frame
    ligand_entries = pd.concat([ligand_entries.reset_index(drop=True),
                                pd.DataFrame(ligand_entries_classes, columns=range(len(cset)))], axis=1)
    ligand_entries.to_csv(output_path / str('ligands_smiles_class_freq_SP_label.csv'), index=False)
    print("DONE!")


if __name__ == "__main__":
    # read the ligands folder path
    if len(sys.argv) > 3:
        ligands_file_path = sys.argv[1]
        output_path = sys.argv[2]
        smiles_col = sys.argv[3]
    else:
        sys.exit("Wrong number of arguments. Seven arguments must be supplied in order to create a vocabulary "
                 "from a smiles list, encode the smiles with it and also store their classes distribution: \n"
                 "  1. ligands_file_path : path to the csv with the ligands smiles to be used;\n"
                 "  2. output_path : path to the folder where the output should be stored, if it does not exists create it;\n"
                 "  3. smiles_col : the name of the column in the ligands_file_path CSV where the ligands smiles are stored.\n"
                 )

    smiles_decompose_encode_to_cover_SP_classes(ligands_file_path, output_path, smiles_col)
