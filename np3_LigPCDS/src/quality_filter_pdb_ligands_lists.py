from pathlib import Path
import sys
import pandas as pd
from Bio.PDB import PDBParser
from statistics import mean
from tqdm import tqdm

def as_float(num):
    try:
        num = float(num)
    except ValueError as e:
        sys.exit("Wrong numeric parameter:\n"+str(e))
    return num

def isNaN(num):
    return num != num

def print_dict(dict_obj):
    for attribute, value in dict_obj.items():
        print('{} : {}'.format(attribute, value))
    print('')

def pdb_avgBFactor(parser, pdbid, db_path):
    # read the pdb file and compute the average bfactor without the heteroatoms (ligands)
    structure = parser.get_structure(pdbid, (db_path / 'pdb' / str('pdb' + pdbid + '.ent')))
    model = list(structure.get_models())[0]
    avgBfactor = mean([a.get_bfactor() for a in model.get_atoms() if not a.get_full_id()[3][0].startswith('H_',0,3)])
    return round(avgBfactor, 3)

# list the entries classes frequency aggregated for the desired and available/valid ligands
def filter_pdb_ligands_list_quality(pdb_list_file, ligands_list_file, db_path, valid_ligands_list_file, bfactor_ratio_max, bfactor_std_max,
                                       min_occupancy_cutoff, allow_missingHeavyAtoms, max_num_disordered):
    bfactor_ratio_max = as_float(bfactor_ratio_max)
    min_occupancy_cutoff = as_float(min_occupancy_cutoff)
    bfactor_std_max = as_float(bfactor_std_max)
    max_num_disordered = as_float(max_num_disordered)
    allow_missingHeavyAtoms = str(allow_missingHeavyAtoms).upper()

    # store the filtering impact
    total_ligs_filter = {}
    total_PDB_filter = {}

    # check inputs
    db_path = Path(db_path)
    if not db_path.exists() or not db_path.is_dir():
        sys.exit("The provided data folder does not exists.")
    pdb_list_file = Path(pdb_list_file)
    if not pdb_list_file.exists() or not pdb_list_file.is_file():
        sys.exit("The provided pdb list file does not exists.")
    ligands_list_file = Path(ligands_list_file)
    if not ligands_list_file.exists() or not ligands_list_file.is_file():
        sys.exit("The provided ligands list file does not exists.")
    valid_ligands_list_file = Path(valid_ligands_list_file)
    if not valid_ligands_list_file.exists() or not valid_ligands_list_file.is_file():
        sys.exit("The provided valid ligands list file does not exists.")

    if allow_missingHeavyAtoms == "TRUE":
        allow_missingHeavyAtoms = True
    else:
        allow_missingHeavyAtoms = False

    # read the pdb list containing the desired pdb codes
    pdb_list = pd.read_csv(pdb_list_file,
                           usecols=['PDBID', 'RefinementResolution', 'SpaceGroup', 'AverageBFactor'])
    pdb_list['PDBID'] = pdb_list.PDBID.str.lower() # pdb id to lower
    total_PDB_filter['total_filtered_PDB_entries'] = pdb_list.shape[0]
    # n = pdb_list.shape[0]
    # read the ligands list containing the desired ligandID by PDB entry
    ligs_list = pd.read_csv(ligands_list_file,
                           usecols=['EntryID', 'LigandID', 'freeLigand', 'LigandFormula','LigandMW','numTotalCount'])
    # create the list of valid ligandCode by entryID
    ligs_list['PDBID'] = ligs_list.EntryID.str.lower()  # pdb id to lower
    ligs_list["ligCode_PDBID"] = ligs_list.LigandID + "_" + ligs_list.PDBID

    print("*** Start filtering the valid ligands SDF that appears in the filtered PDB and ligand entries lists ***\n")
    # read the available and valid ligands list
    valid_ligs_list = pd.read_csv(valid_ligands_list_file,
                            na_values=['null', 'N/A', 'NA'], keep_default_na=False)
    total_ligs_filter['total_filtered_ligs_totalCountByEntry'] = ligs_list.numTotalCount.sum()
    total_ligs_filter['total_unique_filtered_ligs'] = ligs_list.LigandID.unique().size
    total_ligs_filter['total_valid_ligs'] = valid_ligs_list.shape[0]
    total_ligs_filter['total_unique_valid_ligs'] = valid_ligs_list.ligCode.unique().size
    # resolution filter -filter the valid ligands present in the entries of the pdb_list and retrieve the entries info
    valid_ligs_list = valid_ligs_list[valid_ligs_list.entry.isin(pdb_list.PDBID)]
    total_ligs_filter['subtotal_ligs_resolution_depDate'] = valid_ligs_list.shape[0]
    # NP atoms - filter only the valid ligands that appears in the ligands list entries by PDBID
    valid_ligs_list = valid_ligs_list[(valid_ligs_list.ligCode+"_"+valid_ligs_list.entry).isin(ligs_list.ligCode_PDBID)]
    total_ligs_filter['subtotal_ligs_res_depDate_NP_atoms'] = valid_ligs_list.shape[0]
    # set the free ligands using the ligands entries list
    valid_ligs_list['Free_ligand'] = False
    valid_ligs_list.loc[(valid_ligs_list.ligCode + "_" + valid_ligs_list.entry).isin(
	    ligs_list.ligCode_PDBID[ligs_list.freeLigand == True]), "Free_ligand"] = True
    # check if the PDB entries files are present and compute the global entry bfactor if missing
    pdb_list['missing_data'] = False
    pdb_list['missing_ligs'] = False
    # create the PDB parser with quiet True to disable construction warnings (i.e. discontinuous chains) and permissive to allow missing atoms
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    print("*** Start checking the PDB entries files and computing the average B factor of the protein when needed ***\n")
    for i in tqdm(range(pdb_list.shape[0])):
        #if i % 1000 == 0:
        #    print("\n** Processing entry " + str(i) + "/" + str(pdb_list.shape[0]) + " **\n")
        pdbid = pdb_list.PDBID[i]
        #
        # check if .mtz and .pdb exists
        if not (db_path / 'coefficients' / str(pdbid + '.mtz')).exists() or not (
                db_path / 'pdb' / str('pdb' + pdbid + '.ent')).exists():
            # print('ERROR missing entry ' + pdbid + ' files')
            pdb_list.loc[i, 'missing_data'] = True
            continue
        #
        # compute bfactor cutoff
        if isNaN(pdb_list.AverageBFactor[i]) or pdb_list.AverageBFactor[i] <= 0:
            # if the average B-factor is missing in the pdb list
            # read the pdb file and compute the average bfactor without the heteroatoms (ligands)
            pdb_list.loc[i, 'AverageBFactor'] = pdb_avgBFactor(parser, pdbid, db_path)
        #
        # filter available ligands for this entry
        entry_ligs = valid_ligs_list.loc[valid_ligs_list.entry == pdbid]
        #
        # check if there is any ligand left for this structure, if not skip to the next and remove entry
        if entry_ligs.shape[0] == 0:
            # print('Warning: No ligands for entry ' + pdbid)
            pdb_list.loc[i, 'missing_ligs'] = True
            continue

    print("DONE!")
    print("* A total of " + str(pdb_list.missing_data.sum()) + "/" + str(
        total_PDB_filter['total_filtered_PDB_entries']) + " PDB entries data were missing *\n")
    print("* A total of " + str(pdb_list.missing_ligs.sum()) + "/" + str(
        total_PDB_filter['total_filtered_PDB_entries']) + " PDB entries did not have valid ligands *\n")

    # resolution filter - merge the ligands present in the entries of the pdb_list and retrieve the entries info
    valid_ligs_list = valid_ligs_list.merge(pdb_list, left_on='entry', right_on='PDBID')
    valid_ligs_list = valid_ligs_list.merge(ligs_list.loc[~ligs_list.LigandID.duplicated(),["LigandFormula", "LigandMW", "LigandID"]], left_on='ligCode', right_on='LigandID')
    valid_ligs_list = valid_ligs_list.drop(['PDBID', 'LigandID'], axis=1)  # remove not used info
    # remove missing PDB entries and filter the valid PDB entries in the ligs list
    total_PDB_filter['subtotal_PDB_entries_data_present'] = (~pdb_list.missing_data).sum()
    total_PDB_filter['subtotal_PDB_entries_lig_present'] = (~pdb_list.missing_ligs).sum()
    pdb_list = pdb_list[~(pdb_list.missing_data) & ~(pdb_list.missing_ligs)]
    total_PDB_filter['subtotal_PDB_entries_valid'] = pdb_list.shape[0]
    valid_ligs_list = valid_ligs_list[valid_ligs_list.entry.isin(pdb_list.loc[:, 'PDBID'])]
    total_ligs_filter['valid_ligs_res_depDate_NP_atoms_valid_PDB_entries'] = valid_ligs_list.shape[0]
    valid_ligs_list = valid_ligs_list[valid_ligs_list.Free_ligand]
    total_ligs_filter['subtotal_valid_ligs_resDepDateNPatomsValidPDBFreeligs'] = valid_ligs_list.shape[0]
    print("* A total of " + str(valid_ligs_list.shape[0]) + "/" + str(
        total_ligs_filter['total_valid_ligs']) + " free ligands entries are available in the given PDB list *\n")
    total_ligs_filter['subtotal_unique_valid_ligs_resDepDateNPatomsValidPDBFreeligs'] = len(valid_ligs_list.ligCode.unique())
    print("* A total of " + str(total_ligs_filter['subtotal_unique_valid_ligs_resDepDateNPatomsValidPDBFreeligs']) +
          "/" + str(total_ligs_filter['total_unique_valid_ligs']) +
          " unique free ligands code are available in the given PDB list *\n")


    print("*** Start applying the ligands quality filters ***\n")
    # apply the ligands global quality filter cutoff
    occ_filter = (valid_ligs_list.min_occupancy >= min_occupancy_cutoff)
    total_ligs_filter['valid_ligs_Min_occupancy_>='+str(min_occupancy_cutoff)] = [
        valid_ligs_list.ligCode[occ_filter].unique().size,sum(occ_filter)]
    disordered_cutoff_filter = (valid_ligs_list.numDisordered <= max_num_disordered)
    total_ligs_filter['valid_ligs_max_Num_disordered_<='+str(max_num_disordered)] = [
        valid_ligs_list.ligCode[disordered_cutoff_filter].unique().size,sum(disordered_cutoff_filter)]
    bf_std_filter = (valid_ligs_list.bfactor_std <= bfactor_std_max)
    total_ligs_filter['valid_ligs_Bfactor_std_<='+str(bfactor_std_max)] = [
        valid_ligs_list.ligCode[bf_std_filter].unique().size,sum(bf_std_filter)]
    bf_ratio_filter = (valid_ligs_list.bfactor <= valid_ligs_list.AverageBFactor * bfactor_ratio_max)
    total_ligs_filter['valid_ligs_bfactor_ratio_max_<='+str(bfactor_ratio_max)] = [
        valid_ligs_list.ligCode[bf_ratio_filter].unique().size, sum(bf_ratio_filter)]
    missingHA = (~(valid_ligs_list.missingHeavyAtoms) | allow_missingHeavyAtoms)
    total_ligs_filter['valid_ligs_Missing_Heavy_Atoms'] = [valid_ligs_list.ligCode[missingHA].unique().size, sum(missingHA)]
    valid_ligs_list = valid_ligs_list[missingHA & occ_filter & disordered_cutoff_filter & bf_std_filter & bf_ratio_filter]
    valid_ligs_list = valid_ligs_list.reset_index(drop=True)
    total_ligs_filter['Total_final_valid_ligs_quality_filter'] = [valid_ligs_list.ligCode.unique().size, valid_ligs_list.shape[0]]

    # filter only the pdbids present in the ligands list
    pdb_list = pdb_list[pdb_list.PDBID.isin(valid_ligs_list.loc[:,'entry'])]
    pdb_list = pdb_list.reset_index(drop=True)

    print('* Filtered '+str(total_PDB_filter['subtotal_PDB_entries_valid']-pdb_list.shape[0])+
          '/'+str(total_PDB_filter['subtotal_PDB_entries_valid'])+' pdb entries and '+
          str(total_ligs_filter['subtotal_valid_ligs_resDepDateNPatomsValidPDBFreeligs']-total_ligs_filter['Total_final_valid_ligs_quality_filter'][1])+
          '/'+str(total_ligs_filter['subtotal_valid_ligs_resDepDateNPatomsValidPDBFreeligs'])+' ligands entries *\n')

    pdb_list = pdb_list.drop(['missing_data', 'missing_ligs'], axis=1)
    pdb_list.to_csv(str(pdb_list_file.name.replace(".csv", "_filter_") + "bfRatio_" + str(bfactor_ratio_max) +
                        "bfStd_" + str(bfactor_std_max) + "_occ_" + str(min_occupancy_cutoff) +
                        "_missHAtoms_" + str(allow_missingHeavyAtoms) + "_numDisorder_" + str(max_num_disordered) + ".csv"),
                    index=False)
    total_PDB_filter['Total_final_PDB_entries_valid_quality_filter'] = pdb_list.shape[0]

    valid_ligs_list = valid_ligs_list.drop(['Free_ligand', 'missing_data','missing_ligs'], axis=1)  # remove not used info
    valid_ligs_list.to_csv(str(valid_ligands_list_file.name.replace(".csv", "_") + pdb_list_file.name.replace(".csv", "_filter_") +
                         "bfRatio_" + str(bfactor_ratio_max) + "_bfStd_" + str(bfactor_std_max) + "_occ_" +
                        str(min_occupancy_cutoff) + "_missHAtoms_" + str(allow_missingHeavyAtoms) + "_numDisorder_" +
                        str(max_num_disordered) + ".csv"),
                    index=False)
    # print subtotals
    print_dict(total_PDB_filter)
    print_dict(total_ligs_filter)


if __name__ == "__main__":
    # read the entries folder path, pdb file, quality filters
    if len(sys.argv) >= 10:
        pdb_list_file = sys.argv[1]
        ligands_list_file = sys.argv[2]
        db_path = sys.argv[3]
        valid_ligands_list_file = sys.argv[4]
        bfactor_ratio_max = sys.argv[5]
        bfactor_std_max = sys.argv[6]
        min_occupancy_cutoff = sys.argv[7]
        allow_missingHeavyAtoms = sys.argv[8]
        max_num_disordered = sys.argv[9]
    else:
        sys.exit("Wrong number of arguments. Seven arguments must be supplied in order to filter the valid ligands and "
                 "to apply a quality filter in the provided PDB filtered list and available ligands: \n"
                 "  1. pdb_list_file: The path to the CSV file containing the list of filtered PDB entries (result of step 1.2). "
                 "Mandatory columns = 'PDBID', 'RefinementResolution', 'SpaceGroup', 'AverageBFactor';\n"
                 "  2. ligands_list_file: The path to the CSV file containing the list of filtered ligands entries "
                 "present in the filtered PDB entries (result of step 1.2). "
                 "Mandatory columns = 'EntryID', 'LigandID', 'freeLigand',  'LigandFormula', 'LigandMW', 'numTotalCount';\n"
                 "  3. db_path: The path to the data folder where the directories 'pdb' and 'coefficients' are located;\n"
                 "  4. valid_ligands_list_file: The path to the CSV file containing the list of available ligands with a "
                 "valid sdf file and their info (result of step 1.4). "
                 "Mandatory columns: ligID, entry, ligCode, bfactor, min_occupancy, missingHeavyAtoms, numDisordered;\n"
                 "  5. bfactor_ratio_max: The maximum allowed bfactor ratio between a ligand bfactor and its PDB entry bfactor;\n"
                 "  6. bfactor_std_max: The maximum allowed bfactor standard deviation between the ligand atom's bfactor;\n"
                 "  7. min_occupancy_cutoff: The minimum occupancy cutoff to keep a ligand;\n"
                 "  8. allow_missingHeavyAtoms: The missingHeavyAtoms boolean TRUE (1) or FALSE (0) to allow missing "
                 "heavy atoms in the ligands. If FALSE, no ligands entries with missing heavy atoms will be allowed;\n"
                 "  9. max_num_disordered: The maximum numDisordered that a ligand entry is allowed to have. "
                 "\n\nResults: Two tables will be created in the current directory: \n"
                 "  - '<pdb_list_file.name>_filter_bfactor_<bfactor_ratio_max>_occ_<min_occupancy_cutoff>_missHAtoms_"
                 "<allow_missingHeavyAtoms>_numDisorder_<max_num_disordered>.csv' : containing the "
                 "filtered pdb entries that passed the quality criteria;\n"
                 "  - '<valid_ligands_list_file.name>_<pdb_list_file.name>_filter_bfactor_<bfactor_ratio_max>_occ_<min_occupancy_cutoff>_missHAtoms_"
                 "<allow_missingHeavyAtoms>_numDisorder_<max_num_disordered>.csv' : containing the "
                 "valid ligands that passed the quality criteria.\n"
                 )
    
    filter_pdb_ligands_list_quality(pdb_list_file, ligands_list_file, db_path, valid_ligands_list_file,
                                    bfactor_ratio_max, bfactor_std_max, min_occupancy_cutoff, allow_missingHeavyAtoms,
                                    max_num_disordered)
