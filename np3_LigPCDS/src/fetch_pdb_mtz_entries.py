from Bio.PDB import PDBParser
import urllib.request
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

import subprocess

def convert_sf_cif_to_mtz(input_sf_cif_file, output_mtz_file):
    """
    Converts a structure factor from a CIF file to an MTZ file using gemmi cif2mtz.

    Args:
        input_cif_file (str): Path to the input CIF file.
        output_mtz_file (str): Path for the output MTZ file.
    """
    try:
        command = ["gemmi", "cif2mtz", input_sf_cif_file, output_mtz_file]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        #print(f"Conversion successful: {input_cif_file} -> {output_mtz_file}")
        #if result.stdout:
        #    print("STDOUT:", result.stdout)
        #if result.stderr:
        #    print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during MTZ conversion {input_sf_cif_file} -> {output_mtz_file}: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("Error: 'gemmi' command not found. Ensure gemmi is installed and in your PATH.")
        return False
    return True

def convert_cif_to_pdb(input_cif_file, output_pdb_file):
    """
    Converts a structure from a CIF file to a PDB file using gemmi convert.

    Args:
        input_cif_file (str): Path to the input CIF file.
        output_pdb_file (str): Path for the output PDB file.
    """
    try:
        command = ["gemmi", "convert", input_cif_file, output_pdb_file]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        #print(f"Conversion successful: {input_cif_file} -> {output_mtz_file}")
        #if result.stdout:
        #    print("STDOUT:", result.stdout)
        #if result.stderr:
        #    print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during PDB conversion {input_cif_file} -> {output_pdb_file}: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("Error: 'gemmi' command not found. Ensure gemmi is installed and in your PATH.")
        return False
    return True

def retrieve_sf_cif_to_mtz(file_sf_cif, file_mtz, structure_id):
    if not file_mtz.exists():
        try:
            urllib.request.urlretrieve('https://files.rcsb.org/download/' + structure_id + '-sf.cif',
                                       file_sf_cif)
        except:
            print("ERROR retrieving SF CIF " + structure_id)
            return False
        if not file_sf_cif.exists():
            print("ERROR retrieving SF CIF " + structure_id)
            return False
        # convert sf cif to mtz
        return convert_sf_cif_to_mtz(file_sf_cif, file_mtz)
    return True

def retrieve_cif_to_pdb(file_cif, file_pdb, structure_id):
    if not file_pdb.exists():
        try:
            urllib.request.urlretrieve('https://files.rcsb.org/download/' + structure_id + '.cif',
                                       file_cif)
        except:
            print("ERROR retrieving PDB CIF " + structure_id)
            return False
        if not file_cif.exists():
            print("ERROR retrieving PDB CIF " + structure_id)
            return False
        # convert  cif to pdb
        return convert_cif_to_pdb(file_cif, file_pdb)
    return True


def fetch_cif_pdb_mtz(db_path, pdb_ligs_file, n, i_start=0):
    if n <= i_start and not n == 0:
        sys.exit("The provided stop row is not greater than the start row. Wrong range.")

    # check if folder exists, if not create it
    data_folder = Path(db_path)
    if not data_folder.exists() or not data_folder.is_dir():
        print("Warning: The provided data folder do not exists and will be created.")
        Path(data_folder).mkdir(parents=True, exist_ok=True)
    # create the directories pdb_cif, sf_cif, pdb and coefficients if they do not exists yet
    Path(data_folder / 'pdb_cif').mkdir(parents=True, exist_ok=True)
    Path(data_folder / 'sf_cif').mkdir(parents=True, exist_ok=True)
    Path(data_folder / 'pdb').mkdir(parents=True, exist_ok=True)
    Path(data_folder / 'coefficients').mkdir(parents=True, exist_ok=True)

    #pdbl = PDBList()
    # create the PDB parser for checking the retrieve pdb files, QUIET set to True to suppress warnings
    parser = PDBParser(PERMISSIVE=1, QUIET=True)

    # read csv with pdb list to retrieve
    pdb_retrieve = pd.read_csv(pdb_ligs_file, na_values = ['null', 'N/A'],
                               keep_default_na = False) # do not interpret sodium NA as nan
    pdb_retrieve = pdb_retrieve.PDBID
    if n == 0 or n > len(pdb_retrieve):
        n = len(pdb_retrieve)

    n_success = 0
    # retrieve each structure sf.cif and pdb .cif using the pdb code id,
    # convert them to .pdb and .mtz files using gemmi and try to parse the converted pdb and mtz files
    for i in tqdm(range(i_start,n)):
        structure_id = pdb_retrieve[i].lower()
        print("\n********* START "+ structure_id + " ("+str(i+1)+"/"+str(n)+")\n")

        # retrieve the sf.cif and convert to MTZ file if not done yet
        file_sf_cif = Path(data_folder / 'sf_cif' / str(structure_id + '-sf.cif'))
        file_mtz = Path(data_folder / 'coefficients' / str(structure_id + '.mtz'))
        if not retrieve_sf_cif_to_mtz(file_sf_cif, file_mtz, structure_id):
            continue

        # retrieve the .cif and convert to PDB file if not done yet, if fails also remove mtz
        file_pdb = (data_folder / 'pdb' / str('pdb' + structure_id + '.ent'))
        file_pdb_cif = (data_folder / 'pdb_cif' / str(structure_id + '.cif'))
        if not retrieve_cif_to_pdb(file_pdb_cif, file_pdb, structure_id):
            file_mtz.unlink(missing_ok=True)
            file_pdb.unlink(missing_ok=True)
            file_sf_cif.unlink(missing_ok=True)
            file_pdb_cif.unlink(missing_ok=True)
            continue

        # parse structure to check if its possible to get access to the residues
        try:
            structure = parser.get_structure(structure_id, file_pdb)
        except:
            print("ERROR PDB parser")
            file_mtz.unlink(missing_ok=True)
            file_pdb.unlink(missing_ok=True)
            file_sf_cif.unlink(missing_ok=True)
            file_pdb_cif.unlink(missing_ok=True)
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
        sys.exit("Wrong number of arguments. At least two arguments must be supplied in order to retrieve the "
                 "sf.cif and pdb .cif data and convert them to .pdb and .mtz data: \n"
                 "  1. The path to the data folder where the data will be stored. Four folders will be created: "
                 "'sf_cif' to store the structure factors in .cif, 'pdb_cif' to store the PDB structures in .cif "
                 "(default format now), 'pdb' to store the .pdb files and 'coefficients' to store the .mtz files;\n"
                 "  2. The path to the CSV file containing the PDB ids to be retrieved. Mandatory column = 'PDBID';\n"
                 "  3. (optional) The number of the row where the script should start. "
                 "Skip to the given row or start from the beginning;\n"
                 "  4. (optional) The number of the row of the ligands CSV file where the script should stop. "
                 "Stop in the given row or, if missing, stop in the last row.")
    #
    fetch_cif_pdb_mtz(db_path, pdb_ligs_file, n, i_start=0)
