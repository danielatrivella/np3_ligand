from pathlib import Path
import sys
import pandas as pd
from Bio.PDB import PDBParser
from statistics import mean

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
def filter_pdb_ligands_list_quality(pdb_list_file, db_path, ligands_list_file, bfactor_ratio, bfactor_std_cutoff,
                                       occupancy_cutoff, allow_missingHeavyAtoms, num_disordered_cutoff):
    bfactor_ratio = as_float(bfactor_ratio)
    occupancy_cutoff = as_float(occupancy_cutoff)
    bfactor_std_cutoff = as_float(bfactor_std_cutoff)
    num_disordered_cutoff = as_float(num_disordered_cutoff)
    allow_missingHeavyAtoms = str(allow_missingHeavyAtoms).upper()

    # store the filtering impact
    total_ligs_filter = {}
    total_PDB_filter = {}

    # check inputs
    db_path = Path(db_path)
    if not db_path.exists() or not db_path.is_dir():
        sys.exit("The provided data folder do not exists.")
    if not Path(pdb_list_file).exists():
        sys.exit("The provided pdb list file do not exists.")
    if not Path(ligands_list_file).exists():
        sys.exit("The provided ligands list file do not exists.")

    if allow_missingHeavyAtoms == "TRUE":
        allow_missingHeavyAtoms = True
    else:
        allow_missingHeavyAtoms = False

    # read the pdb list containing the desired pdb codes
    pdb_list = pd.read_csv(pdb_list_file,
                           usecols=['PDBID', 'Resolution', 'SpaceGroup',
                                    'AverageBFactor', 'DepDate', 'ligandsID'])
    pdb_list['PDBID'] = [pdb_id.lower() for pdb_id in pdb_list.PDBID]  # pdb id to lower
    total_PDB_filter['total_PDB_entries'] = pdb_list.shape[0]
    # n = pdb_list.shape[0]

    print("*** Start filtering the ligands that appears in the PDB entries list ***\n")
    # read the available and valid ligands list
    ligs_list = pd.read_csv(ligands_list_file,
                            na_values=['null', 'N/A'], keep_default_na=False)
    total_ligs_filter['total_valid_ligs'] = ligs_list.shape[0]
    total_ligs_filter['total_unique_valid_ligs'] = len(ligs_list.ligCode.unique())
    # resolution filter -filter the ligands present in the entries of the pdb_list and retrieve the entries info
    ligs_list = ligs_list[ligs_list.entry.isin(pdb_list.PDBID)]
    total_ligs_filter['subtotal_ligs_resolution_depDate'] = ligs_list.shape[0]
    # NP atoms - filter only the ligands that appears in the PDB entries
    ligs_list = ligs_list[ligs_list.ligCode.isin(set(' '.join(pdb_list.ligandsID).split(' ')))]
    total_ligs_filter['subtotal_ligs_res_depDate_NP_atoms'] = ligs_list.shape[0]
    ligs_list['Free_ligand'] = False
    # check if the PDB entries files are present and compute the global entry bfactor if missing
    pdb_list['missing_data'] = False
    pdb_list['missing_ligs'] = False
    parser = PDBParser(PERMISSIVE=1)
    print("*** Start checking the PDB entries files, computing the average B factor of the protein when needed and applying the free ligands filter ***\n")
    for i in range(pdb_list.shape[0]):
        if i % 1000 == 0:
            print("\n** Processing entry " + str(i) + "/" + str(pdb_list.shape[0]) + " **\n")
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
        entry_ligs = ligs_list.loc[ligs_list.entry == pdb_list.PDBID[i]]
        #
        entry_ligs = entry_ligs[entry_ligs.ligCode.isin(pdb_list.ligandsID[i].split(' '))]
        #
        # check if there is any ligand left for this structure, if not skip to the next and remove entry
        if entry_ligs.shape[0] == 0:
            # print('Warning: No ligands for entry ' + pdbid)
            pdb_list.loc[i, 'missing_ligs'] = True
            continue
        # check the remaining ligands as filtered in the quality assessment
        ligs_list.loc[entry_ligs.index, "Free_ligand"] = True

    print("DONE!")
    print("* A total of " + str(pdb_list.missing_data.sum()) + "/" + str(
        total_PDB_filter['total_PDB_entries']) + " PDB entries data were missing *\n")
    print("* A total of " + str(pdb_list.missing_ligs.sum()) + "/" + str(
        total_PDB_filter['total_PDB_entries']) + " PDB entries did not have valid ligands *\n")

    # resolution filter - merge the ligands present in the entries of the pdb_list and retrieve the entries info
    ligs_list = ligs_list.merge(pdb_list, left_on='entry', right_on='PDBID')
    ligs_list = ligs_list.drop(['PDBID', 'ligandsID'], axis=1)  # remove not used info
    # remove missing PDB entries and filter the valid PDB entries in the ligs list
    total_PDB_filter['subtotal_PDB_entries_data_present'] = pdb_list[pdb_list.missing_data == False].shape[0]
    total_PDB_filter['subtotal_PDB_entries_lig_present'] = pdb_list[pdb_list.missing_ligs == False].shape[0]
    pdb_list = pdb_list[~(pdb_list.missing_data) & ~(pdb_list.missing_ligs)]
    total_PDB_filter['subtotal_PDB_entries_valid'] = pdb_list.shape[0]
    ligs_list = ligs_list[ligs_list.entry.isin(pdb_list.loc[:, 'PDBID'])]
    total_ligs_filter['valid_ligs_res_depDate_NP_atoms_valid_PDB_entries'] = ligs_list.shape[0]
    ligs_list = ligs_list[ligs_list.Free_ligand]
    total_ligs_filter['subtotal_valid_ligs_resDepDateNPatomsValidPDBFreeligs'] = ligs_list.shape[0]
    print("* A total of " + str(ligs_list.shape[0]) + "/" + str(
        total_ligs_filter['total_valid_ligs']) + " free ligands entries are available in the given PDB list *\n")
    total_ligs_filter['subtotal_unique_valid_ligs_resDepDateNPatomsValidPDBFreeligs'] = len(ligs_list.ligCode.unique())
    print("* A total of " + str(total_ligs_filter['subtotal_unique_valid_ligs_resDepDateNPatomsValidPDBFreeligs']) +
          "/" + str(total_ligs_filter['total_unique_valid_ligs']) +
          " unique free ligands code are available in the given PDB list *\n")


    print("*** Start applying the ligands quality filters ***\n")
    # apply the ligands global quality filter cutoff
    occ_filter = (ligs_list.min_occupancy >= occupancy_cutoff)
    total_ligs_filter['valid_ligs_Min_occupancy_>='+str(occupancy_cutoff)] = [
        len(ligs_list.ligCode[occ_filter].unique()),sum(occ_filter)]
    disordered_cutoff_filter = (ligs_list.numDisordered <= num_disordered_cutoff)
    total_ligs_filter['valid_ligs_Num_disordered_cutoff_<='+str(num_disordered_cutoff)] = [
        len(ligs_list.ligCode[disordered_cutoff_filter].unique()),sum(disordered_cutoff_filter)]
    bf_std_filter = (ligs_list.bfactor_std <= bfactor_std_cutoff)
    total_ligs_filter['valid_ligs_Bfactor_std_<='+str(bfactor_std_cutoff)] = [
        len(ligs_list.ligCode[bf_std_filter].unique()),sum(bf_std_filter)]
    bf_ratio_filter = (ligs_list.bfactor <= ligs_list.AverageBFactor * bfactor_ratio)
    total_ligs_filter['valid_ligs_Bfactor_ratio_<='+str(bfactor_ratio)] = [
        len(ligs_list.ligCode[bf_ratio_filter].unique()), sum(bf_ratio_filter)]
    missingHA = (~(ligs_list.missingHeavyAtoms) | allow_missingHeavyAtoms)
    total_ligs_filter['valid_ligs_Missing_Heavy_Atoms'] = [len(ligs_list.ligCode[missingHA].unique()), sum(missingHA)]
    ligs_list = ligs_list[missingHA & occ_filter & disordered_cutoff_filter & bf_std_filter & bf_ratio_filter]
    ligs_list = ligs_list.reset_index()
    total_ligs_filter['Total_final_valid_ligs_quality_filter'] = [len(ligs_list.ligCode.unique()), ligs_list.shape[0]]

    # filter only the pdbids present in the ligands list
    pdb_list = pdb_list[pdb_list.PDBID.isin(ligs_list.loc[:,'entry'])]
    pdb_list = pdb_list.reset_index()

    print('* Filtered '+str(total_PDB_filter['subtotal_PDB_entries_valid']-pdb_list.shape[0])+
          '/'+str(total_PDB_filter['subtotal_PDB_entries_valid'])+' pdb entries and '+
          str(total_ligs_filter['subtotal_valid_ligs_resDepDateNPatomsValidPDBFreeligs']-total_ligs_filter['Total_final_valid_ligs_quality_filter'][1])+
          '/'+str(total_ligs_filter['subtotal_valid_ligs_resDepDateNPatomsValidPDBFreeligs'])+' ligands entries *\n')

    errors_idx_count = []
    print("*** Recompute the ligandsID for each PDB entry and remove entries with no ligand left ***\n")
    for i in range(pdb_list.shape[0]):
        # filter available ligands for this entry
        entry_ligs = ligs_list.loc[ligs_list.entry == pdb_list.PDBID[i]]
        # check if there is any ligand left for this structure, if not skip to the next and remove entry
        if entry_ligs.shape[0] == 0:
            #print('Warning: No ligands for entry ' + pdbid)
            errors_idx_count.append(i)
            continue
        # update valid ligs ids
        pdb_list.loc[i, "ligandsID"] = ' '.join(entry_ligs.ligCode.unique())

    print("DONE!")
    print("* A total of " + str(len(errors_idx_count)) + "/" + str(pdb_list.shape[0]) +
          " PDB entries had no ligands left *\n")
    # remove missing entries and save the filtered pdb entries list
    pdb_list = pdb_list.drop(pdb_list.index[errors_idx_count], axis=0)
    pdb_list = pdb_list.drop(['missing_data', 'missing_ligs'], axis=1)
    pdb_list.to_csv(str(Path(pdb_list_file).name.replace(".csv", "_filter_") + "bfRatio_" + str(bfactor_ratio) +
                        "bfStd_" + str(bfactor_std_cutoff) + "_occ_" + str(occupancy_cutoff) +
                        "_missHAtoms_" + str(allow_missingHeavyAtoms) + "_numDisorder_" + str(num_disordered_cutoff) + ".csv"),
                    index=False)
    total_PDB_filter['Total_final_PDB_entries_valid_quality_filter'] = pdb_list.shape[0]

    ligs_list = ligs_list.drop(['Free_ligand', 'missing_data','missing_ligs'], axis=1)  # remove not used info
    ligs_list.to_csv(str(Path(ligands_list_file).name.replace(".csv", "_") + Path(pdb_list_file).name.replace(".csv", "_filter_") +
                         "bfRatio_" + str(bfactor_ratio) + "_bfStd_" + str(bfactor_std_cutoff) + "_occ_" +
                        str(occupancy_cutoff) + "_missHAtoms_" + str(allow_missingHeavyAtoms) + "_numDisorder_" +
                        str(num_disordered_cutoff) + ".csv"),
                    index=False)
    # print subtotals
    print_dict(total_PDB_filter)
    print_dict(total_ligs_filter)


if __name__ == "__main__":
    # read the entries folder path, pdb file, quality filters
    if len(sys.argv) >= 9:
        pdb_list_file = sys.argv[1]
        db_path = sys.argv[2]+ '/'
        ligands_list_file = sys.argv[3]
        bfactor_ratio = sys.argv[4]
        bfactor_std_cutoff = sys.argv[5]
        occupancy_cutoff = sys.argv[6]
        allow_missingHeavyAtoms = sys.argv[7]
        num_disordered_cutoff = sys.argv[8]
    else:
        sys.exit("Wrong number of arguments. Seven arguments must be supplied in order to filter the free ligands and "
                 "to apply a quality filter in the provided PDB filtered list and available ligands: \n"
                 "  1. pdb_list_file: The path to the CSV file containing the list of filtered PDB entries. "
                 "Mandatory columns = 'PDBID', 'Resolution', 'SpaceGroup', 'AverageBFactor', 'ligandsID';\n"
                 "  2. db_path: The path to the data folder where the directories 'pdb' and 'coefficients' are located;\n "
                 "  3. ligands_list_file: The path to the CSV file containing the list of available ligands with a "
                 "valid sdf file and their info. "
                 "Mandatory columns: ligID, entry, ligCode, bfactor, min_occupancy, missingHeavyAtoms, numDisordered;\n"
                 "  4. bfactor_ratio: The maximum allowed bfactor ratio between a ligand bfactor and its PDB entry bfactor;\n"
                 "  5. bfactor_std: The maximum allowed bfactor standard deviation between the ligand atom's bfactor;\n"
                 "  6. occupancy_cutoff: The minimum occupancy cutoff to keep a ligand;\n"
                 "  7. allow_missingHeavyAtoms: The missingHeavyAtoms boolean TRUE (1) or FALSE (0) to allow missing "
                 "heavy atoms in the ligands. If FALSE, no ligands entries with missing heavy atoms will be allowed;\n"
                 "  8. num_disordered_cutoff: The maximum numDisordered that a ligand entry is allowed to have. "
                 "\nResults: Two tables will be created in the current directory: "
                 "- '<pdb_list_file.name>_filter_bfactor_<bfactor_ratio>_occ_<occupancy_cutoff>_missHAtoms_"
                 "<allow_missingHeavyAtoms>_numDisorder_<num_disordered_cutoff>.csv' : containing the "
                 "filtered pdb entries that passed the quality criteria;\n"
                 "- '<ligands_list_file.name>_<pdb_list_file.name>_filter_bfactor_<bfactor_ratio>_occ_<occupancy_cutoff>_missHAtoms_"
                 "<allow_missingHeavyAtoms>_numDisorder_<num_disordered_cutoff>.csv' : containing the "
                 "ligands that passed the quality criteria.\n"
                 )
    filter_pdb_ligands_list_quality(pdb_list_file, db_path, ligands_list_file, bfactor_ratio, bfactor_std_cutoff,
                                    occupancy_cutoff, allow_missingHeavyAtoms, num_disordered_cutoff)
