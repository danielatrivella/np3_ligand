import sys
from subprocess import PIPE, Popen
from pathlib import Path
from pandas import read_csv, Series, DataFrame
import time, os
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from tqdm import tqdm
from time import strftime, localtime
import shutil

def refine_pdb_entry(entry_row, pdb_path, mtz_path, refinement_path, verbose, pbar):
    i = entry_row[0]
    entryid = entry_row[1]["entryID"]
    entry_refine = entry_row[1]["refinement"]
    no_hetatm = entry_row[1]["noHetatm"]
    if no_hetatm == 1:
        # remove hetatm, set it as the parameter name with spacing
        no_hetatm = "--no-hetatm "
    else:
        # do not remove hetatm, set is as empty
        no_hetatm = ""

    refinement_out = Series({'entryID': entryid, 'error': "False"})
    refinement_out['time_process'] = None
    t1 = time.time()
    # call the refinement of the pdb entry with the id equals entryid
    if verbose:
        print("\n********* START " + entryid + " (" + str(i + 1) + ") *********\n")

    entry_pdb_file = pdb_path / str(entryid + '.pdb')
    entry_mtz_file = mtz_path / str(entryid + '.mtz')
    # check if entry input files exists
    if not entry_pdb_file.exists() or not entry_mtz_file.exists():
        if verbose:
            print("ERROR missing entry " + entryid + " input files")
        refinement_out['error'] = "ERROR missing entry " + entryid + " input files"
        pbar.update(1)
        return refinement_out

    out_dir = refinement_path / entryid
    # if the entry should not be refined (probable already is), only copy the .pdb and .mtz files to the output directory
    if entry_refine == 0:
        if verbose:
            print("Done! Entry",entryid,"is already refined, coping its input pdb and mtz file to the output "
                  "refinement directory.")
        # create the out_dir if it does not exists
        if not out_dir.exists() or not out_dir.is_dir():
            out_dir.mkdir(parents=True)
        shutil.copy(entry_mtz_file.as_posix(), out_dir / entry_mtz_file.name)
        shutil.copy(entry_pdb_file.as_posix(), out_dir / entry_pdb_file.name)
        pbar.update(1)
        return refinement_out
    elif (out_dir / entry_pdb_file.name).exists() and (out_dir / entry_mtz_file.name).exists():
        if verbose:
            print("Done! Entry", entryid, "is already refined, output pdb and mtz files already exists.")
            pbar.update(1)
            return refinement_out

    # run dimple with default parametersand 2xSlow
    # use parameter to disable hetatm '--no-hetatm' when desired
    try:
        p = Popen("dimple " + entry_mtz_file.as_posix() + " " + entry_pdb_file.as_posix() +
                  " " + out_dir.as_posix() + ' --hklout ' + entry_mtz_file.name + " --xyzout " +
                  entryid + ".pdb "+no_hetatm+"-s -s", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        stdout = str(stdout)
        stderr = str(stderr)
        # dimple_stats = stream.read()
        # returncode: A None value indicates that the process has not terminated yet. A negative value -N indicates
        # that the child was terminated by signal N (POSIX only).
        if p.returncode != 0:
            if verbose:
                print("FAILED to refine entry " + entryid + ", process returned code = "+str(p.returncode)+"\n" +
                      ('' if not dimple_stats.stdout else str(dimple_stats.stdout + "\n" + dimple_stats.stderr)))
            # entries_errors.append(entryid)
            # continue
            refinement_out['error'] = "ERROR refining entry " + entryid + ", process returned code = "+str(p.returncode)+\
                                      "\n" + (stdout if stderr == '' else str(stdout + "\nERROR\n" + stderr))
            pbar.update(1)
            return refinement_out
    except:
        if verbose:
            print("ERROR Dimple FAILED to refine entry " + entryid + "\n" +
                  ('' if not dimple_stats.stdout else str(dimple_stats.stdout + "\n" + dimple_stats.stderr)))
        refinement_out['error'] = "ERROR refining entry " + entryid + "\n" + \
                                      (stdout if stderr == '' else str(stdout + "\nERROR\n" + stderr))
        pbar.update(1)
        return refinement_out

    d = time.time() - t1
    # print(p.returncode)
    # print("Processed ligand", lig_name, "in: %.2f s." % d)
    refinement_out['time_process'] = d
    if verbose:
        print("Done in: %.2f s." % d, "! Refinement of entry",entryid, "was successful!")
    pbar.update(1)
    return refinement_out

class EngineRefinePDBentry(object):
    def __init__(self, pdb_path, mtz_path, refinement_path, verbose, pbar):
        self.pdb_path = pdb_path
        self.mtz_path = mtz_path
        self.refinement_path = refinement_path
        self.verbose = verbose
        self.pbar = pbar
    def __call__(self, entry_row):
        # entry_row: iterrows result
        return refine_pdb_entry(entry_row, self.pdb_path, self.mtz_path, self.refinement_path,
                                self.verbose, self.pbar)

def refine_entries(db_path, entry_list_path, num_processors = 2, refinement_path = None):
    if num_processors > cpu_count():
        num_processors = max(cpu_count() - 1, 1)
        print("Warning: The selected number of processors is greater than the available CPUs, setting it to the number of CPUs - 1 = ",
              num_processors)
    # check if the input folders exists
    db_path = Path(db_path)
    if not db_path.exists() or not db_path.is_dir():
        sys.exit("The provided data folder do not exists.")
    pdb_path = db_path / 'pdb'
    if not pdb_path.exists() or not pdb_path.is_dir():
        sys.exit("The 'pdb' folder, with the .pdb files, is no present in the provided data folders.")
    mtz_path = db_path / 'mtz'
    if not mtz_path.exists() or not mtz_path.is_dir():
        sys.exit("The 'mtz' folder, with the .mtz files, is no present in the provided data folders.")

    # check if refinement dir exists, if not create it
    if refinement_path is None:
        refinement_path = db_path / strftime("refinement_%Y%m%d_%Hh%Mm%Ss", localtime())
    else:
        refinement_path = Path(refinement_path)
    if not refinement_path.exists() or not refinement_path.is_dir():
        refinement_path.mkdir()

    # read the input entries to be refined, check mandatory columns ['entryID', 'refinement', 'noHetatm']
    entries_list = read_csv(entry_list_path)
    if not entries_list.columns.isin(['entryID', 'refinement', 'noHetatm']).sum() == 3:
        sys.exit("The 'entry_list_table' CSV file do not have the following mandatory columns:",
                 entries_list.columns[~entries_list.columns.isin(['entryID', 'refinement', 'noHetatm'])].values,
                 "\nRefactor this table and execute the script again.")

    entries_list = entries_list[['entryID', 'refinement', 'noHetatm']]
    #  unique entryID
    entries_list = entries_list[~entries_list.entryID.duplicated()].reset_index(drop=True)
    n = entries_list.shape[0]

    # run dimple for the provided PDB entries, removing heteroatoms
    print("\n**************************************")
    print("*** Running Refinement with Dimple ***")
    print("**************************************\n")
    pbar = tqdm(total=n)
    try:
        pool_refinements = ThreadPool(num_processors)
        # create the engine to create the ligand pc passing the parameters
        # set verbose as True if num_processors is 1 (not parallel)
        engine_refinePDBs = EngineRefinePDBentry(pdb_path, mtz_path, refinement_path, (num_processors == 1), pbar)
        refinements_out = pool_refinements.map(engine_refinePDBs, entries_list.iterrows())
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool_refinements.close()
        pool_refinements.join()
    # convert the result in a dataframe and merge with the entries data
    refinements_out = DataFrame(refinements_out)
    # print(refinements_out)
    print("\n\nDone refining the provided entries using dimple!")
    print("Average time spent to refine the entries: %.2f s." % refinements_out.time_process.mean())

    entries_errors = refinements_out.error.str.match("ERROR")
    if any(entries_errors):
        print("ERROR msgs:", refinements_out.error[entries_errors].values)
        print("\nA total of "+str(sum(entries_errors))+" entries raised an error. The following entries couldn't be "+
              "refined:\n  - "+str(refinements_out.entryID[entries_errors].values).strip('[]'))

    # save refinement out, removing errors
    refinements_out = refinements_out.drop(
        index=refinements_out.index[refinements_out.error.str.match("ERROR")]).reset_index(drop=True)
    refinements_out = refinements_out.merge(entries_list)
    refinements_out.to_csv((refinement_path / "refinement_entries_list.csv").as_posix(), index=False)
    return refinement_path


if __name__ == "__main__":
    # optional refinement path to continue from previous refined data
    num_processors = 2
    refinement_path = None
    # read the db folder path and the PDB list file
    if len(sys.argv) >= 3:
        db_path = sys.argv[1]
        entries_list_path = sys.argv[2]
        if len(sys.argv) > 3:
            num_processors = int(sys.argv[3])
        if len(sys.argv) > 4:
            refinement_path = sys.argv[4]
            if refinement_path.lower() == "none":
                refinement_path = None

    else:
        sys.exit("Wrong number of arguments. Two arguments must be supplied in order to refine a list of entries "
                 "using Dimple: \n"
                 "  1. data_folder: The path to the data folder where the 'pdb' and the 'mtz' folders are located, "
                 "containing the entries files named as <entryID>.pdb and <entryID>.mtz, respectively. "
                 "A folder named 'refinement_<DATE>' will be created to store the dimple results (when applicable) "
                 "organized in subfolders, one for each entry present in the entry_list_table.\n"
                 "  2. entry_list_table: The path to the csv table with one entry by row, defining the datasets that "
                 "will be processed. "
                 "Three mandatory columns: \n    - 'entryID' with a unique entry name for each dataset "
                 "(will be used to retrieve the pdb and mtz data from the data_folder);\n"
                 "    - 'refinement' with 0 or 1 values defining if the given dataset should be refined with Dimple "
                 "(1) or not (0). If 0, the entry's pdb and mtz files present in the data_folder will be copied to the "
                 "entry refinement output directory;\n"
                 "    - and 'noHetatm' with 0 or 1 values defining if the hetero atoms present in the entries .pdb "
                 "files should be removed before the refinement (1 - it can remove already modeled ligands) or not "
                 "(0 - it will keep all ligands, useful to keep waters, clusters and protein co-factors);\n"
                 "  3. parallel_cores: (optional) The number of processors to use for multiprocessing parallelization."
                 " If 1 parallelization is disable and more verbose messages are enabled."
                 "(Default to 2);\n"
                 "  4. refinement_path: (optional) The path to a previous refinement result to continue refining the "
                 "missing entries. Used to continue from a previous result. If it is None a new refinement output "
                 "directory is created. (Default to None)\n"

                 )
    refine_entries(db_path, entries_list_path, num_processors, refinement_path)

