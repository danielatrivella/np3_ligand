# script to organize crystallographic data from fragalysis with PanDDA results into NP3 Blob Label expected files organization
# and the LigPCDS structure labeling input organization

import sys
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
import numpy as np

# copy ligands .pdb and sdf files to the 'ligands' folder using the ligID as name, created inside the output_path folder
# rename the panddas event map and diff map using suffix .ccp4 and _fofc.ccp4 as expected by the np3 blob label app
# check the entries with valid files (diff map, pandda event and ligands structure) and then filter only the valid entries
# create table with columns 'ligID','ligCode','smiles' from fragalysis metadata columns Long code, Compound code and Smiles, respective

def copy_file(source_file, destination_file):
	# copy blob image file
	try:
		shutil.copy(source_file, destination_file)
		#print("File copied successfully.")
	# If source and destination are the same
	except shutil.SameFileError:
		print("Source and destination represents the same file. Source:",source_file,"destination:",destination_file)
		return 1
	# If there is any permission issue
	except PermissionError:
		print("Permission denied. Source:",source_file,"destination:",destination_file)
		return 1
	# For other errors
	except Exception as e:
		print("Error occurred while copying file. Source:",source_file,"destination:",destination_file)
		print(e)
		return 1
	return 0

def organize_fragalysis_data_for_np3_blob_label(metadata_path, aligned_files_path, output_path):
	metadata_path = Path(metadata_path)
	aligned_files_path = Path(aligned_files_path)
	output_path = Path(output_path)
	
	# read the list of ligands to be moved
	metadata = pd.read_csv(metadata_path)
	cols_keep = ['Code', 'Long code', 'Experiment code', 'Compound code', 'Smiles', 'Centroid res', 'Pose', 'RefinementResolution']
	if not metadata.columns.isin(cols_keep).sum() == len(cols_keep):
		sys.exit("ERROR: the list of columns to keep is not present in the metadata table. Missing columns: "+
		         ",".join(np.setdiff1d(metadata.columns[metadata.columns.isin(cols_keep)],cols_keep)))
	# filter only cols of interest
	metadata = metadata.loc[:, cols_keep]
	# create LigPCDS cols, rename kept cols
	metadata['entry'] = metadata.Code
	metadata['ligID'] = metadata['Long code']
	metadata['ligCode'] = metadata['Compound code']
	metadata['smiles'] = metadata['Smiles']
	metadata['error'] = False
	# create the new output path if it does not exist yet
	output_path.mkdir(exist_ok=True, parents=True)
	# create the 'ligands' dir inside it
	ligands_dir_path = output_path / 'ligands'
	ligands_dir_path.mkdir(exist_ok=True, parents=True)
	
	n_missing_pandda_event_data = 0
	n_missing_map_diff_data = 0
	n_missing_ligand_data = 0
	for i in tqdm(range(metadata.shape[0]), desc="Copy fragalysis result to NP3 Blob Label organization"):
		# check if the respective blob image exists in the np3 output path
		entryID = metadata.entry[i]
		pandda_event_file = aligned_files_path / entryID / str(entryID + "_event_crystallographic.ccp4")
		map_diff_file = aligned_files_path / entryID / str(entryID + "_diff_crystallographic.ccp4")
		ligand_sdf_file = aligned_files_path / entryID / str(entryID + "_ligand.sdf")
		ligand_pdb_file =aligned_files_path / entryID / str(entryID + "_ligand.pdb")
		if not pandda_event_file.exists():
			n_missing_pandda_event_data += 1
			metadata.loc[i,"error"] = True
			continue
		if not map_diff_file.exists():
			n_missing_map_diff_data += 1
			metadata.loc[i,"error"] = True
			continue
		if not ligand_sdf_file.exists() or not ligand_pdb_file.exists():
			n_missing_ligand_data += 1
			metadata.loc[i,"error"] = True
			continue
		
		# copy ligands structure files to the ligands dir in the output_path
		cp_res = copy_file(ligand_sdf_file, ligands_dir_path / str(metadata.ligID[i]+"_NO_H.sdf"))
		if cp_res == 1:
			metadata.loc[i,"error"] = True
			n_missing_ligand_data += 1
			continue
		cp_res = copy_file(ligand_pdb_file, ligands_dir_path / str(metadata.ligID[i] + ".pdb"))
		if cp_res == 1:
			metadata.loc[i,"error"] = True
			n_missing_ligand_data += 1
			continue
		
		# rename the maps data
		cp_res = copy_file(pandda_event_file, ligands_dir_path / aligned_files_path / entryID / str(entryID + ".ccp4"))
		if cp_res == 1:
			metadata.loc[i,"error"] = True
			n_missing_pandda_event_data += 1
			continue
		cp_res = copy_file(map_diff_file, ligands_dir_path / aligned_files_path / entryID / str(entryID + "_fofc.ccp4"))
		if cp_res == 1:
			metadata.loc[i,"error"] = True
			n_missing_map_diff_data += 1
			continue
		# create fake mtz
		cp_res = copy_file(map_diff_file, ligands_dir_path / aligned_files_path / entryID / str(entryID + ".mtz"))
		if cp_res == 1:
			metadata.loc[i, "error"] = True
			n_missing_map_diff_data += 1
			continue
		
	# print the final number of correctly copied files and misisng ones
	print("\nTotal of",n_missing_ligand_data, "/", metadata.shape[0], "ligands data (.sdf and .pdb) were missing\n")
	print("Total of", n_missing_map_diff_data, "/", metadata.shape[0], "difference map data were missing\n")
	print("Total of", n_missing_pandda_event_data, "/", metadata.shape[0], "pandda event map were missing\n")
	print("Total of", (~metadata.error).sum(), "/", metadata.shape[0],
	      "entries were correctly processed!!\n")
	metadata = metadata.loc[~metadata.error, :]
	metadata.to_csv(output_path / str("metadata_panddas_fragalysis_data_ok.csv"), index=False)


if __name__ == "__main__":
	import sys
	
	# parse arguments
	if len(sys.argv) >= 4:
		metadata_path = sys.argv[1]
		aligned_files_path = sys.argv[2]
		output_path = sys.argv[3]
	else:
		sys.exit(
			"Wrong number of arguments. Three arguments must be supplied to reorganize a PanDDA fragalysis data to the "
			"NP3 Blob Label input structure and the LigPCDS ligands folder (for structure labeling the data). "
			"This script will copy the ligands structure data to the new output_path/ligands folder "
			"(which may then be used for structure labeling the ligands) and will rename "
			"the PanDDA event map and the diff map present in the aligned_files_path with suffixes .ccp4 and _fofc.ccp4, "
			"respectively. Parameters: \n"
			"  1. metadata_path: The path to the CSV metadata table inside the PanDDA fragalysis dataset folder. "
			"It must contain the following mandatory columns: 'Code', 'Long code', 'Experiment code', 'Compound code', "
			"'Smiles', 'Centroid res', 'Pose', 'RefinementResolution';\n"
			"  2. aligned_files_path: The path to the 'aligned_files' folder inside the PanDDA fragalysis dataset.');\n"
			"  3. output_path: the new output path to store the result. Usually set as the parent dir of the aligned_files_path.\n"
		)
	organize_fragalysis_data_for_np3_blob_label(metadata_path, aligned_files_path, output_path)