from Bio.PDB import PDBList, PDBParser
import urllib.request
import pandas as pd
import sys
from pathlib import Path

def retrieve_mtz(file_mtz, structure_id):
    if not file_mtz.exists():
        try:
            urllib.request.urlretrieve('https://edmaps.rcsb.org/coefficients/' + structure_id + '.mtz',
                                       file_mtz)
        except:
            print("ERROR MTZ " + structure_id)
            return False
        if not file_mtz.exists():
            print("ERROR MTZ " + structure_id)
            return False
    return True

def retrieve_pdb(pdbl, file_pdb, structure_id):
    if not file_pdb.exists():
        try:
            pdbl.retrieve_pdb_file(structure_id, pdir=file_pdb.parent, file_format='pdb')
        except:
            print("ERROR PDB " + structure_id)
            return False
        # if could not download file, return error
        if not file_pdb.exists():
            print("ERROR PDB " + structure_id)
            return False
    return True

def fetch_pdb_mtz(db_path, pdb_ligs_file, n, i_start=0):
    if n <= i_start and not n == 0:
        sys.exit("The provided stop row is not greater than the start row. Wrong range.")

    # check if folder exists, if not create it
    data_folder = Path(db_path)
    if not data_folder.exists() or not data_folder.is_dir():
        print("Warning: The provided data folder do not exists and will be created.")
        Path(data_folder).mkdir(parents=True, exist_ok=True)
    # create the directories pdb and coefficients if they do not exists yet
    Path(data_folder / 'pdb').mkdir(parents=True, exist_ok=True)
    Path(data_folder / 'coefficients').mkdir(parents=True, exist_ok=True)

    pdbl = PDBList()
    parser = PDBParser(PERMISSIVE=1)

    # read csv with pdb list to retrieve
    pdb_retrieve = pd.read_csv(pdb_ligs_file, na_values = ['null', 'N/A'],
                               keep_default_na = False) # do not interpret sodium NA as nan
    pdb_retrieve = pdb_retrieve.PDBID
    if n == 0 or n > len(pdb_retrieve):
        n = len(pdb_retrieve)

    n_success = 0
    # retrieve each structure .pdb and .mtz files using the pdb code id and try to parse the pdb file
    for i in range(i_start,n):
        structure_id = pdb_retrieve[i].lower()
        print("\n********* START "+ structure_id + " ("+str(i+1)+"/"+str(n)+")\n")

        # retrieve the MTZ file if not done yet
        file_mtz = Path(data_folder / 'coefficients' / str(structure_id + '.mtz'))
        if not retrieve_mtz(file_mtz, structure_id):
            continue

        # retrieve the PDB file if not done yet, if fails also remove mtz
        file_pdb = (data_folder / 'pdb' / str('pdb' + structure_id + '.ent'))
        if not retrieve_pdb(pdbl, file_pdb, structure_id):
            file_mtz.unlink(missing_ok=True)
            continue

        # parse structure to check if its possible to get access to the residues
        try:
            structure = parser.get_structure(structure_id, file_pdb)
        except:
            print("ERROR PDB parser")
            file_mtz.unlink(missing_ok=True)
            file_pdb.unlink(missing_ok=True)
            continue

        n_success = n_success + 1
        print("Downloaded ("+str(n_success)+"/"+str(i+1)+")\n")

    print("Successfully downloaded "+str(n_success)+" of "+str(n-i_start)+" structures and reflections data!")

if __name__ == "__main__":
    i_start = 0
    n = 0
    if len(sys.argv) >= 3:
        db_path = sys.argv[1]
        pdb_ligs_file = sys.argv[2]
        if len(sys.argv) > 3:
            i_start = int(sys.argv[3])
        if len(sys.argv) > 4:
            n = int(sys.argv[4])
    else:
        sys.exit("Wrong number of arguments. At least two arguments must be supplied in order to retrieve the .pdb and .mtz data: \n"
                 "  1. The path to the data folder where the data will be stored. Two folders will be created: 'pdb' to store "
                 "te .pdb files and 'coefficients' to store the .mtz files;\n"
                 "  2. The path to the CSV file containing the PDB ids to be retrieved. Mandatory column = 'PDBID';\n"
                 "  3. (optional) The number of the row where the script should start. "
                 "Skip to the given row or start from the beginning;\n"
                 "  4. (optional) The number of the row of the ligands CSV file where the script should stop. "
                 "Stop in the given row or, if missing, stop in the last row.")

    fetch_pdb_mtz(db_path, pdb_ligs_file, n, i_start=0)
