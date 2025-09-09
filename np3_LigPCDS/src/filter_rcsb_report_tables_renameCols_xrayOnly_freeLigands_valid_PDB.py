# script to filter from the RCSB PDB lists the entries with experimental method equals x-ray diffraction and rename the
# original columns
# Then, filter only the ligands that appear as a free ligand in the PDB entry
# first from the ligands lists search the PDB entries in which each ligand ID appear as a free ligand and mark the free ligand entries
# then, filter from the PDB entries the ligands that are covalent (not free) and remove any PDB entry without a free ligand
# return the two new filtered lists

import urllib.request
import json
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import http
from itertools import chain

def filter_xrayOnly_renameCols(report_PDB_path, report_Ligands_path, output_path):
    output_path = Path(output_path)
    if not output_path.exists() or not output_path.is_dir():
        print("Warning: The provided output folder do not exists and will be created.")
        output_path.mkdir(parents=True, exist_ok=True)
    print("Number of entries before filtering:")
    report_PDB_list = pd.read_csv(report_PDB_path)
    # refactor the colum names removing spaces, units between parenthesis and the Length word
    report_PDB_list.columns = report_PDB_list.columns.str.replace(" ", "").str.replace("\(.*\)", "",
                                                                                       regex=True).str.replace("Length",
                                                                                                               "")
    print("  - PDB entries: ", report_PDB_list.shape[0])
    # filter only experiments from pure X-ray diffraction
    report_PDB_list = report_PDB_list.loc[report_PDB_list.ExperimentalMethod == "X-RAY DIFFRACTION",
                      :].reset_index(drop=True)
    # Remove columns where ALL values are NaN
    report_PDB_list = report_PDB_list.dropna(axis=1, how='all')
    # save the new PDB table
    output_PDB_list_path = output_path/"report_PDB_rcsb_pdb_2008-02-01_protein_xrayOnly_hasLigand_hasExpData.csv"
    report_PDB_list.to_csv(output_PDB_list_path, index=False)
    # filter the kept PDBs in the ligands list
    report_Ligands_list = pd.read_csv(report_Ligands_path)
    # first remove spaces from the column names
    report_Ligands_list.columns = report_Ligands_list.columns.str.replace(" ", "")
    print("  - Ligand entries: ", report_Ligands_list.shape[0])
    print("** Filtering the PDB IDs and respective ligands with experimental methods equals only X-Ray Diffraction **")
    # then filter the valid PDBs
    report_Ligands_list = report_Ligands_list.loc[report_Ligands_list.EntryID.isin(report_PDB_list.EntryID),
                          :].reset_index(drop=True)
    # next remove the ligands IDs with NaN values
    report_Ligands_list = report_Ligands_list.loc[~report_Ligands_list.LigandID.isna(),:]
    # Remove columns where ALL values are NaN
    report_Ligands_list = report_Ligands_list.dropna(axis=1, how='all')
    # save the new Ligands table
    output_Ligands_list_path = output_path/"report_Ligands_rcsb_pdb_2008-02-01_protein_xrayOnly_hasLigand_hasExpData.csv"
    report_Ligands_list.to_csv(output_Ligands_list_path, index=False)
    print("Number of entries after filtering X-Ray Diffraction experimental method only:")
    print("  - PDB entries: ", report_PDB_list.shape[0])
    print("  - Ligand entries: ", report_Ligands_list.shape[0])
    # return the output lists names
    return output_PDB_list_path, output_Ligands_list_path


def filter_freeLigands_validPDB_rcsbAPI(report_PDB_path, report_Ligands_path, output_path):
    if not report_PDB_path.exists():
        sys.exit("The provided path to the PDB report table do not exists. Stopped filtering the RCSB PDB reports.")
    if not report_Ligands_path.exists():
        sys.exit("The provided path to the Ligands report table do not exists. Stopped filtering the RCSB PDB reports.")

    report_Ligands_list = pd.read_csv(report_Ligands_path)

    # rcsb pdb query api to search for the PDB entries in which the <ligand_ID> appears as a free ligand
    # replace the <ligand_ID> with the desired ligand ID to send the request
    query_rcsb_pdb_with_freelig = 'https://search.rcsb.org/rcsbsearch/v2/query?json=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_nonpolymer_instance_annotation.comp_id%22%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22<ligand_ID>%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_nonpolymer_instance_annotation.type%22%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22HAS_NO_COVALENT_LINKAGE%22%7D%7D%5D%7D%2C%22return_type%22%3A%22entry%22%2C%22request_options%22%3A%7B%22results_verbosity%22%3A%22compact%22%2C%22return_all_hits%22%3Atrue%7D%7D'

    print("** Retrieving the free ligand information from RCSB PDB API **")
    # for each unique ligand present in the report ligands list, search the PDBIDS in which they appear as a free ligand
    # and mark the result in the table, default to not free
    report_Ligands_list["freeLigand"] = False
    #for ligandID in tqdm(report_Ligands_list.loc[~report_Ligands_list.freeLigand,:].LigandID.unique()):  # for only the not free ligands
    for ligandID in tqdm(report_Ligands_list.LigandID.unique()):
        print("Ligand ID: ", ligandID)
        # build the query with the ligID
        query_rcsb_pdb_with_freelig_ligID = query_rcsb_pdb_with_freelig.replace("<ligand_ID>", ligandID)
        # send the query request
        req = urllib.request.Request(query_rcsb_pdb_with_freelig_ligID, headers={'Content-Type': 'application/json'})
        # parse the query result
        try:
            with urllib.request.urlopen(req) as response:
                response_body = response.read().decode('utf-8')
        except (urllib.error.URLError, http.client.RemoteDisconnected, urllib.error.HTTPError) as e:
            response_body = None
            print(f"Error: {e.reason}")
        # parse the response
        if response_body:
            pdbIDs_list = json.loads(response_body)['result_set']
        else:
            if response_body is not None:
                # response body is empty - no error
                print(f"There is no PDB ID where the ligand ID equals '{ligandID}' appears as a Free ligand.")
            else:
                print(f"Could not retrieve any PDB IDs where the ligand ID equals '{ligandID}' appears as a Free ligand.")
            continue
        # check if there is a valid pdb
        if (len(pdbIDs_list) == 0):
            print(f"There is no PDB ID where the ligand ID equals '{ligandID}' appears as a Free ligand.")
            continue
        # filter in the ligands list the ligandID that appear in the pdbIDs_list and set it as free ligand
        report_Ligands_list.loc[((report_Ligands_list.LigandID == ligandID) &
                                 (report_Ligands_list.EntryID.isin(pdbIDs_list))), "freeLigand"] = True
    #
    # store the complete table with the freeLigands signalized
    report_Ligands_list.to_csv(output_path/"report_Ligands_rcsb_pdb_2008-02-01_protein_xrayOnly_hasLigandFree_hasExpData.csv",
                               index=False)
    print(f"Total number of free ligands by PDBID: {report_Ligands_list.freeLigand.sum()}")
    print(f"Total number of unique free ligands (unique ligand ID): {report_Ligands_list.loc[report_Ligands_list.freeLigand, 'LigandID'].unique().size}.")
    # filter the free ligands to compute their count by pdb id
    report_Ligands_list = report_Ligands_list.loc[report_Ligands_list.freeLigand == True,:]
    # read PDB lists and for each entry compute the number of free ligands
    report_PDB_list = pd.read_csv(report_PDB_path)
    # for each PDB entry, compute the number of free ligands and update in the PDB table
    # add columns NumberofDistinctFreeLigands and TotalNumberofFreeLigands
    report_PDB_list["NumberofDistinctFreeLigands"] = 0
    report_PDB_list["TotalNumberofFreeLigands"] = 0
    for i in range(report_PDB_list.shape[0]):
        pdbid = report_PDB_list.loc[i,"PDBID"]
        ligands_pdbid = report_Ligands_list.loc[report_Ligands_list.EntryID == pdbid,:]
        num_distinctFreeLigands = ligands_pdbid.shape[0]
        # if no free ligand, leave counts as zero and proceed to next ligandID
        if num_distinctFreeLigands == 0:
            continue
        asym_freeLigs = list(chain.from_iterable([asymID.split(",")
                                                  for asymID in ligands_pdbid.AsymID
                                                  if asymID != "" and not pd.isna(asymID)]))
        authasym_freeLigs = list(chain.from_iterable([authasymID.split(",")
                                                      for authasymID in ligands_pdbid.AuthAsymID
                                                      if authasymID != "" and not pd.isna(authasymID)]))
        num_totalFreeLigands = len(asym_freeLigs) + len(authasym_freeLigs)
        report_PDB_list.loc[i, "NumberofDistinctFreeLigands"] = num_distinctFreeLigands
        report_PDB_list.loc[i, "TotalNumberofFreeLigands"] = num_totalFreeLigands
    # store pdb list with count of free ligands
    print(f"Total number of free ligands by PDBID in different chains: {report_PDB_list.TotalNumberofFreeLigands.sum()}")
    report_PDB_list.to_csv(output_path/"report_PDB_rcsb_pdb_2008-02-01_protein_xrayOnly_hasLigandFree_hasExpData.csv", index=False)

def filter_rcsb_report_tables_xray_renameCols_freeLigand_validPDBs(report_PDB_path, report_Ligands_path, output_path):
    # call the filtering functions
    report_PDB_filtered_path, report_Ligands_filtered_path =  filter_xrayOnly_renameCols(report_PDB_path, report_Ligands_path, output_path)
    filter_freeLigands_validPDB_rcsbAPI(report_PDB_filtered_path, report_Ligands_filtered_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        report_PDB_path = sys.argv[1]
        report_Ligands_path = sys.argv[2]
        output_path = sys.argv[3]
    else:
        sys.exit("Wrong number of arguments. Three argument are necessary to filter the report tables from RCSB PDB "
                 "(concatenated in previous step) to only keep data from pure X-Ray Experiments, "
                 "to rename the columns removing spaces and parenthesis, and to retrieve from the RCSB PDB API the free "
                 "ligands information (long process) to enrich the reports. "
                 "Four tables will be created in the output_path as a result of "
                 "filtering and enriching the PDB and the Ligands reports from RCSB PDB. List of parameters: \n"
                 "  1. report_PDB_path: The path to the PDB report table downloaded from RCSB PDB and concatenated in previous step (CSV format). It must contain structure data;\n"
                 "  2. report_Ligands_path: The path to the Ligands report table downloaded from RCSB PDB and concatenated in previous step (CSV format). It must contain ligand (non-polymer entity) data;\n"
                 "  3. output_path: The path to the output directory where the resulting filtered and enriched tables will be stored.\n"
                 "\nResulting tables that will be created inside the output_path:\n"
                 "- report_PDB_rcsb_pdb_2008-02-01_protein_xrayOnly_hasLigand_hasExpData.csv : The PDB report table with only X-Ray experimental method entries filtered and columns renamed without spaces \n"
                 "- report_Ligands_rcsb_pdb_2008-02-01_protein_xrayOnly_hasLigand_hasExpData.csv : The Ligands report table with only X-Ray experimental method entries filtered and columns renamed without spaces \n"
                 "- report_Ligands_rcsb_pdb_2008-02-01_protein_xrayOnly_hasLigandFree_hasExpData.csv : The Ligands report table filtered with only X-Ray exp and Free ligands information retrieved and signalized in a new column 'freeLigand' with True or False values \n"
                 "- report_PDB_rcsb_pdb_2008-02-01_protein_xrayOnly_hasLigandFree_hasExpData.csv: The PDB report table filtered with only X-Ray exp and Free ligands counts in new columns 'NumberofDistinctFreeLigands' and 'TotalNumberofFreeLigands'\n"
                 )
    filter_rcsb_report_tables_xray_renameCols_freeLigand_validPDBs(report_PDB_path, report_Ligands_path, output_path)