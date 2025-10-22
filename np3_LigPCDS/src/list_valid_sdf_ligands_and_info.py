from pathlib import Path
import sys
import pandas as pd
import rdkit.Chem as chem
from rdkit import RDLogger
from chemutils import get_smiles, add_CNOFH_charges
from Bio.PDB import MMCIFParser
from statistics import mean
import numpy as np
from tqdm import tqdm

def lig_cif_name(file_name, pdb_suffix='_lig.cif'):
	file_name = file_name.split('_')
	return str(file_name[0]+'_'+file_name[1]+'_'+file_name[3]+'_'+file_name[2]+pdb_suffix)

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
	# prevent undesired warning msgs that are frequent with the sdf data, such as
	# # Warning: molecule is tagged as 2D, but at least one Z coordinate is not zero. Marking the mol as 3D.
	RDLogger.DisableLog('rdApp.warning')
	# check inputs
	# print(db_ligand_path)
	db_ligand_path = Path(db_ligand_path)
	if not db_ligand_path.exists() or not db_ligand_path.is_dir():
		sys.exit("The provided data folder do not exists.")

	# CIF-PDB parser to retrieve the ligands occupancy and bfactor by atom
	parser = MMCIFParser(QUIET=True)
	#parser = PDBParser(PERMISSIVE=1)

	# store the valid ligands sdf present in the db path informations: b factor
	ligand_entries = []
	errors_count = []
	print("*** Start listing and validating ligands ***")
	db_ligs_sdf_path = list(db_ligand_path.glob("*_NO_H.sdf"))
	for sdf in tqdm(db_ligs_sdf_path):
		#print(sdf.name)
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
		if not (db_ligand_path / lig_cif_name(sdf.name)).exists():
			print('ERROR the ligand cif-pdb file '+ lig_cif_name(sdf.name)+ ' does not exists.')
			errors_count.append(sdf.name)
			continue
		structure = parser.get_structure(sdf.name[0:-9], db_ligand_path / lig_cif_name(sdf.name))
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
	
	if ligand_entries.shape[0] < len(db_ligs_sdf_path):
		print("Finish. There were", len(db_ligs_sdf_path)-ligand_entries.shape[0], "ligand sdf files with error.")
	print("DONE! A total of ", ligand_entries.shape[0], "ligand sdf files were correctly validated!")
	# store valid ligands infos
	ligand_entries.to_csv(db_ligand_path.name+'_valid_sdf_info.csv', index=False)
	
	# from the given ligands table
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
				 "\n\nResult: One table will be created in the current directory named: "
				 "- ligands_data_folder.name+'_valid_sdf_info.csv': containing the list of available ligands with a valid SDF file and their information."
				 )
	list_valid_sdf_ligands_and_info(db_ligand_path)
