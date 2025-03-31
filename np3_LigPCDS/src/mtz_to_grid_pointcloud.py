import gemmi
from pathlib import Path
import numpy as np
import open3d as o3d
import sys
from shutil import rmtree
import pandas as pd
import time
from p_tqdm import p_map
from multiprocessing import cpu_count

def std_mean_median_q25th_q75th(x):
    return np.std(x), np.mean(x), np.median(x), np.quantile(x, 0.25), np.quantile(x, 0.75)


def min_max(x):
    return np.min(x), np.max(x)


def create_PDB_ligs_grid_pc(pdb_entry, lig_data_out, refinement_data_path, output_path, overwrite_pc, grid_space,
                            sample_rate, verbose=False):
    if verbose:
        print("\n*** Start processing PDB entry", pdb_entry, "***")
    t1 = time.time()
    n = lig_data_out.shape[0]
    lig_data_out.loc[:,"point_cloud"] = False
    lig_data_out.loc[:,"rho_mean"] = -1
    lig_data_out.loc[:,"rho_std"] = -1
    lig_data_out.loc[:,"rho_median"] = -1
    lig_data_out.loc[:,"rho_q25th"] = -1
    lig_data_out.loc[:,"rho_q75th"] = -1
    lig_data_out.loc[:,"missing_ligand"] = "False"
    lig_data_out.loc[:,"time_process"] = -1
    #
    mtz_file = refinement_data_path / pdb_entry / (pdb_entry + ".mtz")
    if not Path(mtz_file).is_file():
        if verbose:
            print("Warning: The PDB entry ", pdb_entry,
              " was not refined successfully. All its ligands will be skipped.")
        lig_data_out.loc[:,"missing_ligand"] = "PDBRefinementError"
        return lig_data_out[['ligID', 'point_cloud', 'rho_mean', 'rho_std', 'rho_median', 'rho_q25th', 'rho_q75th',
                             'missing_ligand', 'time_process']]
    #
    # validate lig output path
    lig_output_path = output_path / pdb_entry
    if not lig_output_path.exists():
        lig_output_path.mkdir(parents=True)
    elif overwrite_pc and lig_output_path.exists():
        rmtree(lig_output_path, ignore_errors=True)
        lig_output_path.mkdir(parents=True)
    # create the pdb entry density map
    try:
        # read mtz file
        mtz = gemmi.read_mtz_file(mtz_file.as_posix())
        # mtz.nreflections   # from MTZ record NCOL
        # mtz.min_1_d2       # from RESO
        # mtz.max_1_d2       # from RESO
        # mtz.resolution_low()   # sqrt(1 / min_1_d2)
        # mtz.resolution_high() # sqrt(1 / max_1_d2)
        # mtz.datasets
        #
        # define grid space or sample rate to interpolate density grid values
        if not sample_rate:
            # grid_space was informed
            # define the sample rate to interpolate density grid values
            sample_rate = mtz.resolution_high() / grid_space  # = dmin/grid_size
        else:
            # round the grid_space to facilitate further quantization
            grid_space = np.round(mtz.resolution_high() / sample_rate, 2)
            sample_rate = mtz.resolution_high() / grid_space
        # set the used grid_space in the current ligand
        lig_data_out.loc[:, "grid_space"] = grid_space

        if verbose:
            print("Resolution high = ", mtz.resolution_high(), "\nGrid space = ", grid_space,
              "\nSample rate = ", sample_rate)
        # mtz columns after REFMAC - https://www.globalphasing.com/buster/wiki/index.cgi?MTZcolumns
        # 2Fo-Fc map coefficients ->  amplitude: FWT pahses: PHWT
        # Fo-Fc (difference map) coefficients -> amplitude: DELFWT pahses: PHDELWT
        # Figure of merit -> FOM
        # Model structure factor -> FC_ALL PHIC_ALL : Atomic and bulk-solvent model;
        # Reflection indices -> H,K,L -> no scale
        # Observed data -> FP,SIGFP (or similar) : (amplitude and sigma). The column names will be identical to the column names of the input data -> scale Observational (unmodified from input)
        # Flag indicating which reflections are in the free set -> FreeR_flag : Copied from input if present, otherwise created afresh.
        # Calculated structure factors -> FC,PHIC
        #
        # obtain map 2Fo-Fc in direct space xyz
        # map_grid_2fofc = mtz.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=sample_rate)
        # obtain diff map Fo-Fc in direct space xyz
        # use 1.5 times the final sample rate to increase accuracy of the linear interpolation
        map_grid_fofc = mtz.transform_f_phi_to_map('DELFWT', 'PHDELWT', sample_rate=sample_rate*1.2)
        del mtz
        #
        # compute sigma contour
        if verbose:
            print("Computing the Electron Density mean,std,median,quantile25th,quantile75th")
        # converting the map to a numpy array
        std_I, mean_I, median_I, quantile25th_I, quantile75th_I = std_mean_median_q25th_q75th(np.array(map_grid_fofc,
                                                                                                     copy=False))
        lig_data_out.loc[:,"rho_mean"] = mean_I
        lig_data_out.loc[:,"rho_std"] = std_I
        lig_data_out.loc[:,"rho_median"] = median_I
        lig_data_out.loc[:,"rho_q25th"] = quantile25th_I
        lig_data_out.loc[:,"rho_q75th"] = quantile75th_I
        # contour_sigma = [mean_I+std_I*-sigma_factor,mean_I+std_I*sigma_factor]
        if verbose:
            print("rho Standard deviation = ", std_I, "\nrho mean = ", mean_I, "\nrho median = ", median_I,
              "\nrho quantile25th = ", quantile25th_I, "\nrho quantile75th = ", quantile75th_I)
        # "\nContour sigma = ", contour_sigma)
        #
        # apply symmetry TODO check if it is really necessary - removed for now
        # map_grid_fofc.symmetrize_max()
        # std_I, mean_I = std_mean(np.array(map_grid_fofc, copy=False))
        # map_grid_2fofc.symmetrize_max()
    except Exception as e:
        if verbose:
            print("Warning: Could not read the mtz file ", mtz_file.name, "skipping PDB entry.")
            print("Error message:", e)
        lig_data_out.loc[:,"missing_ligand"] = "MtzParsingError"
        return lig_data_out[['ligID', 'point_cloud', 'rho_mean', 'rho_std', 'rho_median', 'rho_q25th', 'rho_q75th',
                             'missing_ligand', 'time_process']]
    #
    d = time.time() - t1
    if verbose:
        print("Processed PDB entry", pdb_entry, "in: %.2f s." % d)
    lig_data_out.loc[:,"time_process"] = d / n
    #
    # for each ligand in the current entry, interpolate its grid inside the defined bounding box in the
    # fo-fc electron density map and then extract its grid point cloud
    for i in range(n):
        # get the ligand id
        lig_name = lig_data_out.loc[i, 'ligID']
        if (lig_output_path / (lig_name+"_grid_point_cloud_fofc.xyzrgb")).exists() and not overwrite_pc:
            if verbose:
                print("* Already processed ligand", lig_name, "-", i+1,"/", n,". Skipping to next entry. *")
            lig_data_out.loc[i, "point_cloud"] = True
            # pbar.update(1)
            continue
        if verbose:
            print("\n* Start processing ligand entry", lig_name, "-", i+1,"/", n,"*")
        t1 = time.time()
        # lig_label_pos_file = db_ligxyz_path / (lig_name+'_class.xyz')
        # get the ligand center position xyz and the box bound size
        xyz = lig_data_out.loc[i, ['x','y','z']].values.astype(np.float64)
        xyz_bound = lig_data_out.loc[i, ['x_bound','y_bound','z_bound']].values.astype(np.float64)
        # shift to a position equals the <scale_boundingbox> times the bound size from the ligand center and
        # create a 3d grid 2*<scale_boundingbox> times the original bounding box size
        # the position xyz0 is the 0,0,0 point of the 3d grid
        scale_boundingbox = 1.2
        xyz0 = (xyz-scale_boundingbox/2*xyz_bound).round()
        grid_size = (xyz_bound*(scale_boundingbox)/grid_space).astype(int)
        if verbose:
            print("Center x,y,z = ",xyz,"\nbounding box size lx,ly,lz = ", xyz_bound,
              "\nScale = ",scale_boundingbox,
              "\nGrid size = ", grid_size)
            #
            print("- Interpolate the density map values in the ligand bounding box grid")
        try:
            # first we create a numpy array of the same type as the grid, to store the density values
            fofc_3d_grid = np.zeros(grid_size, dtype=np.float32)
            # twofofc_3d_grid = np.zeros(grid_size, dtype=np.float32)
            # then we setup a transformation (array indices) -> (position [A]).
            # the transformation matrix receives the grid_spacing and starts in the vec position xyz0
            # then the function interpolates all the density values inside the grid
            tr = gemmi.Transform()
            tr.mat.fromlist([[grid_space, 0, 0], [0, grid_space, 0], [0, 0, grid_space]])
            tr.vec.fromlist(xyz0)
            # interpolate the density values
            map_grid_fofc.interpolate_values(fofc_3d_grid, tr)
            # map_grid_2fofc.interpolate_values(twofofc_3d_grid, tr)
            # example of point transformation to get the grid indexes of the original point
            # pos_xyz_arr = ((xyz-xyz0)/grid_space).round().astype(int)
            #
            if verbose:
                print("- Create the ligand bounding box grid in a point cloud file")
            # create a point cloud with all the grid points to store the original xyz positions and values
            point_cloud_lig_grid_fofc = ""
            for ijk, v in np.ndenumerate(fofc_3d_grid):
                # transform the grid indices to the real space of the cristal
                xyz_real = (np.array(ijk) * grid_space + xyz0)
                point_cloud_lig_grid_fofc += ' '.join(np.char.mod('%.1f', xyz_real)) + \
                                           (" " + str(fofc_3d_grid[ijk])) * 3 +"\n" #+" 0 "+str(fofc_3d_grid[l, j, k])+"\n" #+ " 0 0\n"
            del fofc_3d_grid #,twofofc_3d_grid
            # save in the xyzrgb format:
            # Each line contains [x, y, z, r, g, b], where r, g, b are in floats of range [0, 1]
            open(lig_output_path / (lig_name+"_grid_point_cloud_fofc.xyzrgb"), "w").write(point_cloud_lig_grid_fofc)
            # read point cloud fofc and set the ligand point cloud column to True
            lig_grid = o3d.io.read_point_cloud(
                (lig_output_path / (lig_name + "_grid_point_cloud_fofc.xyzrgb")).as_posix())
            if verbose:
                print("Number of points in the ligand grid", len(lig_grid.points))
            lig_data_out.loc[i, "point_cloud"] = True
            del lig_grid
        except Exception as e:
            if verbose:
                print("Error interpolating the ligand density values and creating the grid point cloud. Ligand ID:",
                  lig_name, "\nError msg:", e)
            lig_data_out.loc[i, "missing_ligand"] = "GridCreationError"
            # error_ligands.append(lig_name+"_GridInterpolation")
            continue
        #
        d = time.time() - t1
        if verbose:
            print("Processed ligand",lig_name, "in: %.2f s." % d)
        lig_data_out.loc[i, "time_process"] = lig_data_out.loc[i, "time_process"] + d
    #
    return lig_data_out[['ligID', 'point_cloud', 'grid_space',
                         'rho_mean', 'rho_std', 'rho_median', 'rho_q25th', 'rho_q75th',
                         'missing_ligand', 'time_process']]

class EngineLigandGridPc(object):
    def __init__(self, ligs_retrieve, refinement_data_path, output_path,
                 overwrite_pc, grid_space, sample_rate,num_processors):
        self.ligs_retrieve = ligs_retrieve
        self.refinement_data_path = refinement_data_path
        self.output_path = output_path
        self.overwrite_pc = overwrite_pc
        self.grid_space = grid_space
        self.sample_rate = sample_rate
        self.verbose = (num_processors == 1)
    def __call__(self, pdb_entry):
        lig_data_out = self.ligs_retrieve.loc[self.ligs_retrieve.entry == pdb_entry, :].reset_index(drop=True)
        return create_PDB_ligs_grid_pc(pdb_entry, lig_data_out, self.refinement_data_path,
                                       self.output_path, self.overwrite_pc, self.grid_space,
                                       self.sample_rate, self.verbose)

def create_ligands_grid_point_cloud(db_ligxyz_path, refinement_data_path, output_path,
                                    overwrite_pc=False,grid_space=0.2, sample_rate=None,num_processors=2):
    t0 = time.time()
    if num_processors > cpu_count():
        num_processors = max(cpu_count() - 1, 1)
        print("Warning: The selected number of processors is greater than the available CPUs, setting it to the number of CPUs - 1 = ",
              num_processors)

    # check inputs and create missing directories
    refinement_data_path = Path(refinement_data_path)
    db_ligxyz_path = Path(db_ligxyz_path)
    valid_ligs_file = db_ligxyz_path / (db_ligxyz_path.name.replace("xyz_", "") + '_box_class_freq.csv')

    if not db_ligxyz_path.exists() or not db_ligxyz_path.is_dir():
        sys.exit("The provided ligand xyz data folder do not exists.")

    if not refinement_data_path.exists() or not refinement_data_path.is_dir():
        sys.exit("The provided refinement data folder do not exists.")

    if not Path(valid_ligs_file).is_file():
        sys.exit("The provided valid ligands list CSV file do not exists.")

    output_path = Path(output_path)
    if not output_path.exists() or not output_path.is_dir():
        output_path.mkdir(parents=True)

    # read csv with the valid ligands list to encode
    ligs_retrieve = pd.read_csv(valid_ligs_file, na_values = ['null', 'N/A'], keep_default_na = False,
                                usecols = ['ligID', 'ligCode', 'entry', 'Resolution', 'filter_quality', 'x', 'y', 'z',
                                           'x_bound', 'y_bound', 'z_bound']) # do not interpret sodium NA as nan
    # filter out the low quality ligands
    ligs_retrieve = ligs_retrieve[ligs_retrieve.filter_quality]
    n = ligs_retrieve.shape[0]
    # sort by pdb entry name, to read each mtz only once and process all its ligands
    ligs_retrieve = ligs_retrieve.sort_values('entry').reset_index(drop=True)

    ligs_retrieve["point_cloud"] = False

    # for each PDB entry create its ligands grids
    # run the ligand grid pc creation in multiprocessing
    # create the engine to create the ligand pc passing the parameters
    engine_createLigsGridPc = EngineLigandGridPc(ligs_retrieve, refinement_data_path,
                                                 output_path, overwrite_pc, grid_space, sample_rate, num_processors)
    ligands_grid_out = p_map(engine_createLigsGridPc.__call__, ligs_retrieve.entry.unique(),
                             num_cpus=num_processors)
    # convert the result in a dataframe and merge with the ligands data
    ligands_out = pd.concat(ligands_grid_out)
    print("Average time spent to process ligands: %.2f s." % ligands_out.time_process.mean())
    ligands_out = ligands_out.drop('time_process', axis=1)
    ligs_retrieve = ligs_retrieve.drop('point_cloud', axis=1)
    ligs_retrieve = ligs_retrieve.merge(ligands_out, on='ligID')

    # print missing entries and skipped ligands and error ligands
    print("\n")
    entries_missing = ligs_retrieve.missing_ligand.str.match("MtzParsingError|PDBRefinementError")
    if sum(entries_missing) > 0:
        print("- Missing PDB entries:")
        print(ligs_retrieve.entry[entries_missing].unique())
        print("- Skipped ligands due to missing PDB entries:")
        print(ligs_retrieve.ligID[entries_missing].values)

    error_ligands = ligs_retrieve.missing_ligand.str.match("GridCreationError")
    if sum(error_ligands) > 0:
        print("- Ligands that raised an error in the grid creation step:")
        print(ligs_retrieve.ligID[error_ligands].values)

    if sum(entries_missing) > 0:
        print("A total of "+str(ligs_retrieve.entry[entries_missing].unique().shape[0])+"/",
              str(len(ligs_retrieve.entry.unique())),
              " PDB entries refinement were missing.")
        print("A total of "+str(ligs_retrieve.ligID[entries_missing].values.shape[0])+"/", str(n),
              " ligands entries were skipped due to missing PDB entries.")

    if sum(error_ligands) > 0:
        print("A total of "+str(ligs_retrieve.ligID[error_ligands].shape[0])+"/", str(n),
              " ligands entries raised an error and could not be entirely processed.")

    if sum(entries_missing) > 0 or sum(error_ligands) > 0:
        print("Final total " + str(ligs_retrieve.entry[~ligs_retrieve["point_cloud"]].unique().shape[0]) + "/",
              str(len(ligs_retrieve.entry.unique())),
              " PDB entries could not be entirely processed.")
        print("Final total " + str((~ligs_retrieve["point_cloud"]).sum()) + "/", str(n),
              " ligands entries could not be entirely processed.")
    else:
        print("All ligand's grids were successfully created!")
        print("Total PDB entries: ", str(len(ligs_retrieve.entry.unique())))
        print("Total ligand entries: ", str(n))

    # remove failed ligands, save csv of the created point clouds and print number of pdb and ligand errors
    ligs_retrieve = ligs_retrieve[ligs_retrieve["point_cloud"]]
    ligs_retrieve.to_csv((db_ligxyz_path / valid_ligs_file.name.replace("_class_freq.csv", "_pc.csv")).as_posix(),
                         index=False)

    d = time.time() - t0
    print("\nDONE!", "Time elapsed: %.2f s." % d)


if __name__ == "__main__":
    # read the ligands csv with respective pdb entries, x, y, z and data
    overwrite_pc = False
    num_processors = 2
    grid_space = 0.5
    sample_rate = None

    if len(sys.argv) >= 4:
        db_ligxyz_path = sys.argv[1]
        refinement_data_path = sys.argv[2]
        output_path = sys.argv[3]
        if len(sys.argv) >= 5:
            overwrite_pc = (sys.argv[4].lower() == "true")
        if len(sys.argv) >= 6:
            num_processors = int(sys.argv[5])
        if len(sys.argv) >= 7:
            grid_space = sys.argv[6].lower()
            if grid_space == "false":
                grid_space = None
            else:
                grid_space = float(grid_space)
        # not using sample rate, just using grid space
        # if len(sys.argv) >= 8:
        #     sample_rate = sys.argv[7].lower()
        #     if sample_rate == "false":
        #         sample_rate = None
        #     else:
        #         sample_rate = float(sample_rate)
    else:
        sys.exit("Wrong number of arguments. Three parameters must be supplied in order to read the ligands "
                 "Fo-Fc map (mtz file), their atomic positions (xyz file) and the desired output path. "
                 "It will extract the ligand grid image from the refined Fo-Fc map of its PDB entry, using a grid spacing "
                 "equal to 0.5 by default (parameter grid_space). Then, the image will be stored in a point cloud file "
                 "inside each ligand's PDB entry subfolder in the output path (parameter output_grid_path).\nList of parameters: \n"
                 "  1. xyz_labels_path: The path to the data folder called 'ligands/xyz_<valid ligand list csv name>' where the ligands .xyz files with "
                 "their atomic positions and labels are located. It must also contain the CSV file with the valid ligands list and their grid sizing and "
                 "position. This file must be named as '<valid ligand list csv name>_box_class_freq.csv' and is expected to be the output "
                 "of the 'run_vocabulary_encode_ligands.py' script. "
                 "Mandatory columns = 'ligID', 'ligCode', 'entry', 'filter_quality', 'x', 'y', 'z', 'x_bound', 'y_bound',"
                 "'z_bound';\n"
                 "  2. refinement_path: The path to the data folder where the PDB entries refinement are located ('data/refinement');\n"
                 "  3. output_grid_path: The path to the output folder where the point cloud of the ligands' grid "
                 "image will be stored in .xyzrgb files. It will be organized by the PDB entry ID of the ligands in "
                 "separated subfolders, each one containing the grid image of all ligands that appear in that "
                 "entry ('data/ligands_grid_point_clouds');\n"
                 "  4. overwrite: (optional) A boolean True or False indicating if the already processed ligands should be "
                 "overwritten. Useful to restart from previous processing. (Default to False);\n"
                 "  5. num_parallel: (optional) The number of processors to use for multiprocessing parallelization (Default to 2);\n"
                 "  6. grid_space: (optional) A numeric defining the grid spacing size in angstroms to be used in "
                 "the point clouds creation for the ligands' grid image (Default to 0.5 A).\n"
                 )
    print("Parms:")
    print("grid_space", grid_space)
    # print("sample_rate", sample_rate)
    create_ligands_grid_point_cloud(db_ligxyz_path, refinement_data_path, output_path, overwrite_pc,
                                    grid_space, sample_rate, num_processors)
