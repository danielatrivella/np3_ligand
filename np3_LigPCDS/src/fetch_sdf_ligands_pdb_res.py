from Bio.PDB import *
import urllib.request
import pandas as pd
import sys
from pathlib import Path
from rdkit.Chem import SDMolSupplier
from tqdm import tqdm
from rdkit import RDLogger

# Disable the warning message - prevent warning molecule is tagged as 2D, here its always 3D
RDLogger.DisableLog('rdApp.*')

class ResSkip(Select):
    def __init__(self, res, chain):
        self.res = res # store the res id
        self.chain = chain # store the res chain
    def accept_residue(self, residue):
        if residue.get_id()==self.res and residue.get_parent().get_id() == self.chain: # the id of the residue
            #print('Skipping res '+residue.get_resname()+ ' from chain '+self.chain)
            return False
        else:
            return True

class ResSelect(Select):
    def __init__(self, res, chain):
        self.res = res # store the res id
        self.chain = chain # store the res chain
        self.is_water = (self.res[0] == 'W') # store if it is a water residue
    def accept_residue(self, residue):
        # match the id and chain of the residue, if it is a water save all waters together
        if (self.is_water and residue.get_id()[0] == 'W') or \
                (residue.get_id()==self.res and
                 residue.get_parent().get_id() == self.chain):
            #print('Adding res '+residue.get_resname()+' from chain '+self.chain)
            return True
        else:
            return False

def retrieve_sdf(file_sdf, entry, chain, seq):
    if not file_sdf.exists():
        try:
            urllib.request.urlretrieve('https://models.rcsb.org/v1/'+entry+'/ligand?auth_asym_id='+chain+
                                       '&auth_seq_id='+seq+'&encoding=sdf',
                                       file_sdf)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print(f"ERROR retrieving SDF {file_sdf.name}. An unexpected error occurred: {e}")
            return False
        if not file_sdf.exists():
            print("ERROR retrieving SDF " + file_sdf.name)
            return False
    return True



def fetch_sdf_ligands(db_path, ligs_report_file, n, i_start=0):
    if n <= i_start and not n == 0:
        sys.exit("The provided stop row is not greater than the start row. Wrong range.")

    # check if folder exists, if not create it
    data_folder = Path(db_path)
    if not data_folder.exists() or not data_folder.is_dir():
        sys.exit("The provided data folder do not exists.")
    # create the directories ligands and pdb_no_lig if they do not exists yet
    Path(data_folder / 'ligands').mkdir(parents=True, exist_ok=True)
    Path(data_folder / 'pdb_no_lig').mkdir(parents=True, exist_ok=True)
    # create the PDB parser, QUIET set to True to suppress warnings
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    # read csv with ligands list to retrieve
    ligs_retrieve_list = pd.read_csv(ligs_report_file, na_values=['null', 'N/A'], keep_default_na=False)
    # get the list of pdb ids in lower case, to be used to retrieve the ligands data
    ligs_retrieve_list['PDBID'] = ligs_retrieve_list.EntryID.str.lower()

    pdb_list = ligs_retrieve_list.PDBID.unique()

    if n == 0 or n > pdb_list.size:
        n = pdb_list.size

    # for each pdb id, read its structure, extract the ligands from its residues (res), retrieve their sdf files and
    # save the pdb structure without the residues
    for i in tqdm(range(i_start,n)):
        structure_id = pdb_list[i]
        print("\n********* START "+ structure_id + " ("+str(i+1)+"/"+str(n)+")\n")

        # check if the mtz files could be downloaded
        if not Path(data_folder / 'coefficients' / str(structure_id + '.mtz')).exists():
            print("No MTZ found. Skipping structure.")
            continue

        # get the list of already downloaded ligands - check their .pdb file
        ligs_present = list(dict.fromkeys([lig_pdb.name.split('_')[1] for lig_pdb in
                                         data_folder.glob('ligands/'+structure_id+'_*.pdb')]))

        # filter from the ligands report list the ones that appear in this pdbid
        ligs_pdbid_list = ligs_retrieve_list.loc[ligs_retrieve_list.PDBID == structure_id, :]

        # remove the ligands that were already retrieved
        ligs_pdbid_list = ligs_pdbid_list.loc[~ligs_pdbid_list.LigandID.isin(ligs_present),:]
        if ligs_pdbid_list.shape[0] == 0:
            print("Already downloaded all the ligands for this structure. Skipping...")
            continue

        # format the ligandID in the residue naming 'H_<ligandID>'
        ligs_pdbid_list.loc[:, "LigandIDRes"] = ['H_{:>3s}'.format(ligID) for ligID in ligs_pdbid_list.LigandID]

        # parse structure to get access to the residues and ligands
        file_pdb = (data_folder / 'pdb' / str('pdb' + structure_id + '.ent'))
        # check if the pdb files could be downloaded
        if not file_pdb.exists():
            print("No PDB found. Skipping structure.")
            continue
        try:
            structure = parser.get_structure(structure_id, file_pdb)
        except:
            print("ERROR PDB parser")
            continue

        io = PDBIO()
        io.set_structure(structure)
        models = list(structure.get_models())
        if (len(models) == 0):
            print("ERROR no model in PDB file")
            continue

        # iterate over the structure residues in all chains (in AuthAsymID) to locate the desired ligands
        print("** Start parsing ligands and retrieving their SDF data **")
        for residue in models[0].get_residues():
            if residue.get_id()[0] in ligs_pdbid_list.LigandIDRes.values or residue.get_id()[0] == 'W':
                id = list(residue.get_id())
                id[1] = str(id[1])
                id[2] = residue.get_parent().get_id()
                #print(str(id))
                # do not remove the prefix H_ from the water residue and skip its sdf retrieval
                if id[0] == 'W':
                    # water ligs, store all water's residue in a single pdb
                    # save a pdb with all the water coordinates
                    io.save((data_folder / 'ligands' / str(structure_id + "_W_lig.pdb")).as_posix(),
                            ResSelect(residue.get_id(), id[2]))
                else:  # other ligs
                    id_name = '_'.join(id)[2:].strip()
                    # try to download the ligand sdf and then to load it, if it fails remove the sdf file
                    file_sdf = (data_folder / 'ligands' / str(structure_id + "_" + residue.get_resname().strip() + "_" +
                                                              id[2] + "_" + id[1] + '_NO_H.sdf'))
                    if not file_sdf.exists() and retrieve_sdf(file_sdf, structure_id, id[2], id[1]):
                        mol_res = SDMolSupplier(file_sdf.as_posix(), removeHs=True)
                        # if sanitize(mol_res[0]) is None:
                        if mol_res[0] is None:  # without sanitize in the download to try to fix later
                            print('\nERROR parsing sdf ' + file_sdf.name + "\n")
                            file_sdf.unlink(missing_ok=True)
                        elif mol_res[0].GetNumAtoms() == 0:
                            print('ERROR no atoms in sdf ' + file_sdf.name + "\n")
                            file_sdf.unlink(missing_ok=True)
                        else:
                            # if sdf is present
                            # save the ligand in .pdb and the entry pdb without the current ligand
                            if not (data_folder / 'ligands' / str(structure_id + "_" + id_name + "_lig.pdb")).exists():
                                # save a pdb without the ligand residue
                                io.save((data_folder / 'pdb_no_lig' / str(structure_id + "_" + id_name + ".pdb")).as_posix(),
                                        ResSkip(residue.get_id(), id[2]))
                                # save a pdb of the ligand residue, the ligand coordinates
                                io.save((data_folder / 'ligands' / str(structure_id + "_" + id_name + "_lig.pdb")).as_posix(),
                                        ResSelect(residue.get_id(), id[2]))


if __name__ == "__main__":
    i_start = 0
    n = 0
    if len(sys.argv) >= 3:
        db_path = sys.argv[1]+ '/'
        ligs_report_file = sys.argv[2]
        if len(sys.argv) > 3:
            i_start = int(sys.argv[3])
        if len(sys.argv) > 4:
            n = int(sys.argv[4])
    else:
        sys.exit("Wrong number of arguments. At least two arguments must be supplied in order to retrieve the ligands "
                 ".sdf data, extract the ligands .pdb and to create the structures .pdb without each ligand coordinates: \n"
                 "  1. The path to the data folder where the data will be stored. Two folders will be created: 'ligands' to store "
                 "te .sdf and the .pdb files of the ligands and 'pdb_no_lig' to store the .pdb files without the ligand coordinates;\n"
                 "  2. The path to the CSV file containing the ligands report list to be retrieved and the PDB ids in "
                 "which they appear. Mandatory columns = 'EntryID', 'LigandID', and 'AuthAsymID'. The last "
                 "column must contain the chains where the respective LigandID is located in the respective EntryID;\n"
                 "  3. (optional) The number of the row of the ligands CSV file where the script should start. "
                 "Skip to the given row or, if missing, start from the beginning;\n"
                 "  4. (optional) The number of the row of the ligands CSV file where the script should stop. "
                 "Stop in the given row or, if missing, stop in the last row.")
    fetch_sdf_ligands(db_path, ligs_report_file, n, i_start)

