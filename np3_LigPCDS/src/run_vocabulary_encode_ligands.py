from create_vocabulary_valid_ligands_smiles import create_vocabulary_ligand_smiles
from encode_ligs_xyz import encode_ligands_vocabulary
from pathlib import Path
import subprocess,shlex

if __name__ == "__main__":
    import sys

    i_start = 0
    n = 0
    label_SN = True
    # read the entries csv with respective ligands ids
    if len(sys.argv) > 3:
        db_path = sys.argv[1]
        valid_ligs_file = sys.argv[2]
        label_SN = (sys.argv[3].lower() == "true")
        if len(sys.argv) > 4:
            i_start = int(sys.argv[4])
        if len(sys.argv) > 5:
            n = int(sys.argv[5])
    else:
        sys.exit("Wrong number of arguments. Two parameters must be supplied in order to read the ligands "
                 ".sdf data, retrieve their SMILES and create a vocabulary from their smiles labeling. "
                 "Then, this vocabulary will be used to label the ligands' atoms. The ligand grid sizing is also computed here. "
                 "List of parameters: \n"
                 "  1. data_folder_path: The path to the data folder where the vocabulary output will be stored and where the 'ligands' folder "
                 "with the ligands in .sdf format is located. Two files will be created inside it: \n"
                 "- 'ligs_smiles_<ligands_list_path.name>.txt' containing all the smiles used in the vocabulary creation (the smiles database) and;\n"        
                 "- 'vocabulary_<ligands_list_path.name>.txt' containing all the classes that resulted from the smiles labeling, with one class by row (the vocabulary itself).\n"
                 "And one folder called 'xyz_<ligands_list_path>' will be created inside the <data_folder_path>/ligands/ folder to store the labeled ligands, it will contain:\n"
                 "- One .xyz file for each ligand entry ID (ligand + pdb entry) present in the ligands_list_path file;\n"
                 "- One CSV file named '<ligands_list_path>_box_class_freq.csv' containing the list of valid ligands sucessufully labeled, plus their bounding box sizing and vocabulary classes frequency (number of labeled atoms by class);\n"
                 "  2. ligands_list_path: The path to the CSV file containing the valid ligands list and their smiles. This file is expected to be the output of the quality filter script."
                 " Mandatory columns = 'ligID','smiles'.\n"
                 "The name of this file will be used to label the output vocabulary file, the ligands SMILES database file "
                 "and the xyz folder that will store the labeled ligands .xyz files;\n"
                 "  3. label_SP: (optional) Set to 'True' to use the atoms' SP hybridization to create the vocabulary (default), otherwise it will use the atoms' symbol. "
                 "Both labeling approaches will be concatenated with the atoms' cyclic information.\n"
                 "  4. start: (optional) The number of the row in the ligands_list_path file where the script should start. "
                 "Skip to the given row or, if missing, start from the beginning;\n"
                 "  5. end: (optional) The number of the row in the ligands_list_path file where the script should stop. "
                 "Stop in the given row or, if missing, stop in the last row.")

    print("\n***************************************")
    print("*** Running the Vocabulary Creation ***")
    print("***************************************\n")
    create_vocabulary_ligand_smiles(db_path, valid_ligs_file, label_SN)

    job_name = Path(valid_ligs_file).name.replace(".csv", "_"+("SPclasses" if label_SN else "atomSymbol"))
    db_lig_path = db_path + "/ligands/"
    vocab_path = db_path + "/vocabulary_" + job_name + ".txt"
    print("\n************************************")
    print("*** Running the Ligands Encoding ***")
    print("************************************\n")
    encode_ligands_vocabulary(db_lig_path, valid_ligs_file, vocab_path, label_SN, n, i_start)

    # run the statistics over the vocabulary classes coverage
    # print("\n******************************************************")
    # print("*** Running Vocabulary Classes Coverage Statistics ***")
    # print("******************************************************\n")
    # db_xyz_path = db_lig_path + "xyz_" + job_name + "/"
    # vocab_stats = subprocess.run(shlex.split("Rscript src/vocabulary_statistics.R " + db_xyz_path + job_name +
    #                                          "_box_class_freq.csv " + db_xyz_path + "vocabulary_" + job_name + "_freq.csv"))
    # if (vocab_stats.returncode == 1):
    #     print("FAILED to create the vocabulary statisticst")
    #     quit(1)

