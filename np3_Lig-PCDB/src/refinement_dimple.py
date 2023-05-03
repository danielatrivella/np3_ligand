import sys
from subprocess import PIPE, Popen
from pathlib import Path
from pandas import read_csv, Series, DataFrame
import time, os
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from tqdm import tqdm

def refine_pdb_entry(pdbid_i, pdb_path, mtz_path, refinement_path, overwrite_refinement, pbar):
    i = pdbid_i[0]
    pdbid = pdbid_i[1].lower()
    refinement_out = Series({'PDBID': pdbid, 'error': "False"})
    refinement_out['time_process'] = None
    t1 = time.time()
    # call the refinement of the pdb entry with the id equals pdbid
    # print("\n********* START " + pdbid + " (" + str(i + 1) + "/" + str(n) + ")\n")

    entry_pdb_file = pdb_path / str("pdb" + pdbid + '.ent')
    entry_mtz_file = mtz_path / str(pdbid + '.mtz')
    # check if entry input files exists
    if not entry_pdb_file.exists() or not entry_mtz_file.exists():
        # print("ERROR missing entry " + pdbid + " input files")
        refinement_out['error'] = "ERROR missing entry " + pdbid + " input files"
        pbar.update(1)
        return refinement_out

    out_dir = refinement_path / pdbid
    if out_dir.exists() and (out_dir / entry_mtz_file.name).exists() and not overwrite_refinement:
        # print("Output dir already exists. Skipping entry " + pdbid + "...")
        pbar.update(1)
        return refinement_out

    # run dimple with default parameters, no-hetatm and 2xSlow
    try:
        # dimple_stats = subprocess.run(
        #     shlex.split("dimple " + entry_mtz_file.as_posix() + " " + entry_pdb_file.as_posix() +
        #                 " " + out_dir.as_posix() + ' --hklout ' + entry_mtz_file.name + " --xyzout " +
        #                 pdbid + ".pdb --no-hetatm -s -s"), capture_output=True)
        # stream = os.popen("dimple " + entry_mtz_file.as_posix() + " " + entry_pdb_file.as_posix() +
        #                 " " + out_dir.as_posix() + ' --hklout ' + entry_mtz_file.name + " --xyzout " +
        #                 pdbid + ".pdb --no-hetatm -s -s")
        p = Popen("dimple " + entry_mtz_file.as_posix() + " " + entry_pdb_file.as_posix() +
                        " " + out_dir.as_posix() + ' --hklout ' + entry_mtz_file.name + " --xyzout " +
                        pdbid + ".pdb --no-hetatm -s -s", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        stdout = str(stdout)
        stderr = str(stderr)
        # dimple_stats = stream.read()
        if (p.returncode == 1):
            # print("FAILED to refine entry " + pdbid + "\n" +
            #       ('' if not dimple_stats.stdout else str(dimple_stats.stdout + "\n" + dimple_stats.stderr)))
            # entries_errors.append(pdbid)
            # continue
            refinement_out['error'] = "ERROR refining entry " + pdbid + "\n" + \
                                      (stdout if stderr == '' else str(stdout + "\nERROR\n" + stderr))
    except:
        # print("ERROR Dimple FAILED to refine entry " + pdbid + "\n" +
        #       ('' if not dimple_stats.stdout else str(dimple_stats.stdout + "\n" + dimple_stats.stderr)))
        refinement_out['error'] = "ERROR refining entry " + pdbid + "\n" + \
                                      (stdout if stderr == '' else str(stdout + "\nERROR\n" + stderr))

    d = time.time() - t1
    # print("Processed ligand", lig_name, "in: %.2f s." % d)
    refinement_out['time_process'] = d
    pbar.update(1)
    return refinement_out

class EngineRefinePDBentry(object):
    def __init__(self, pdb_path, mtz_path, refinement_path, overwrite_refinement, pbar):
        self.pdb_path = pdb_path
        self.mtz_path = mtz_path
        self.refinement_path = refinement_path
        self.overwrite_refinement = overwrite_refinement
        self.pbar = pbar
    def __call__(self, pdbid_i):
        # pdbid_i: enumarated pdbids
        return refine_pdb_entry(pdbid_i, self.pdb_path, self.mtz_path, self.refinement_path, self.overwrite_refinement,
                                self.pbar)

def refine_pdb_entries(db_path, pdb_list_path, num_processors = 2, overwrite_refinement=False):
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
        sys.exit("The 'pdb' folder, with the .ent (.pdb) files, is no present in the provided data folders.")
    mtz_path = db_path / 'coefficients'
    if not mtz_path.exists() or not mtz_path.is_dir():
        sys.exit("The 'coefficients' folder, with the .mtz files, is no present in the provided data folders.")

    # check if refinement dir exists, if not create it
    refinement_path = db_path / 'refinement'
    if not refinement_path.exists() or not refinement_path.is_dir():
        refinement_path.mkdir()

    # read the input entries to be refined, only select the PDBID column
    pdb_list = read_csv(pdb_list_path, usecols=['PDBID'])
    n = pdb_list.shape[0]
    pdb_list.PDBID = pdb_list.PDBID.str.lower()

    # run dimple for the provided PDB entries, removing heteroatoms
    print("\n**************************************")
    print("*** Running Refinement with Dimple ***")
    print("**************************************\n")
    pbar = tqdm(total=n)
    try:
        pool_refinements = ThreadPool(num_processors)
        # create the engine to create the ligand pc passing the parameters
        engine_refinePDBs = EngineRefinePDBentry(pdb_path, mtz_path, refinement_path, overwrite_refinement, pbar)
        refinements_out = pool_refinements.map(engine_refinePDBs, enumerate(pdb_list.PDBID))
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool_refinements.close()
        pool_refinements.join()
    # convert the result in a dataframe and merge with the ligands data
    refinements_out = DataFrame(refinements_out)
    print(refinements_out)
    print("Average time spent to refine the PDB entries: %.2f s." % refinements_out.time_process.mean())
    refinements_out = refinements_out.drop('time_process', axis=1)

    print("\nDone refining the provided PDB entries using dimple!\n")
    entries_errors = refinements_out.error.str.match("ERROR")
    if any(entries_errors):
        print("ERROR msgs:", refinements_out.error[entries_errors].values)
        print("\nA total of "+str(sum(entries_errors))+" PDB entries raised an error. The following entries couldn't be "+
              "refined:\n  - "+str(refinements_out.PDBID[entries_errors].values).strip('[]'))


if __name__ == "__main__":
    num_processors = 2
    overwrite_refinement = False
    # read the db folder path and the PDB list file
    if len(sys.argv) >= 3:
        db_path = sys.argv[1]
        pdb_list_path = sys.argv[2]
        if len(sys.argv) > 3:
            num_processors = int(sys.argv[3])
        if len(sys.argv) > 4:
            overwrite_refinement = (sys.argv[4].lower() == "true")
    else:
        sys.exit("Wrong number of arguments. Two arguments must be supplied in order to refine a list of PDB entries "
                 "using Dimple (2x slow and no hetero atom): \n"
                 "  1. data_folder_path: The path to the data folder where the 'pdb' and the 'coefficients' folders are "
                 "located, containing the files of the PDB entries named as 'pdb<PDBID>.ent' and '<PDBID>.mtz', "
                 "respectively. A folder named 'refinement' will be created inside the data_folder_path to store the "
                 "Dimple results in separated subfolders by PDB entry;\n"
                 "  2. pdb_list_path: The path to a csv with the PDB list to be refined. Mandatory column: 'PDBID' "
                 "with the PDB entries IDs (they will be converted to lower case);\n"
                 "  3. num_parallel: (optional) The number of processors to use for multiprocessing parallelization "
                 "(default to 2);\n"
                 "  4. overwrite: (optional) A boolean True or False indicating if the already refined entries should "
                 "be overwritten (True) or skipped (False). Default to False.\n"
                 )
    refine_pdb_entries(db_path, pdb_list_path, num_processors, overwrite_refinement)

