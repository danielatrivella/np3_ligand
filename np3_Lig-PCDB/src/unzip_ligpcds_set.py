import pandas as pd
import zipfile
import sys
#

# Extract the LigPCDS entries that are listed in the ligs_set_table within column "entry",
# all ligands of these entries will be extracted to the destination path
# the zip in ligpcds_zip_path must contain the PDB entries names in the subsubfolders (depth=2)
def extract_ligpcds_set(ligpcds_zip_path, ligs_set_table_path, extract_destination_path=""):
    ligs_table = pd.read_csv(ligs_set_table_path)
    entries_list = ligs_table.entry.unique()
    print("* Extracting LigPCDS entries that are present in the provided ligands table within column entry *")
    print("A total of",entries_list.size, "PDB entries will be extracted")
    with zipfile.ZipFile(ligpcds_zip_path) as ligpcds_archive:
        for file in ligpcds_archive.namelist():
            pdb_entry = file.split("/")[1]
            if pdb_entry in entries_list:
                ligpcds_archive.extract(file, extract_destination_path)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        ligpcds_zip_path = sys.argv[1]
        ligs_set_table_path = sys.argv[2]
        extract_destination_path = sys.argv[3]
    else:
        sys.exit("Wrong number of parameters. There are three mandatory parameters to extract LigPCDS dataset from a "
                 "list of PDB entries present in a ligands table (the valid list of ligands table may be used to "
                 "extract all entries present in the dataset and the stratified training table may be used to extract "
                 "only the list of PDB entries present in the training set).\n"
                 "List of parameters:\n"
                 "1. ligpcds_zip_path: The path to the LigPCDS zip folder, inside the LigPCDS-SP or Atom_record dataset. "
                 "It should contain a subfolder named with prefix 'ligands_point_clouds_imgs_' which contains "
                 "subfolders named with the PDB entries names (PDB entries folders in depth = 2);\n"
                 "2. ligs_set_table_path: The path to the table containing the list of ligands and their PDB entries "
                 "to be extracted from ligpcds_zip_path. This table must contain a column named 'entries' with the "
                 "PDB entries to be extracted from the provided LigPCDS dataset.;\n"
                 "3. extract_destination_path: The path to where to extract the selected PDB entries.\n")

    extract_ligpcds_set(ligpcds_zip_path, ligs_set_table_path, extract_destination_path)