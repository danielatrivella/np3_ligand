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
import logging


# passing the map here can multiple the memory used to store this data?
def create_blob_grid_pc(map_grid_file, blob_data, output_path, overwrite_pc, grid_space, verbose=False):
    # for each blob in the current entry, interpolate its grid inside the defined bounding box in the
    # fo-fc electron density map and then extract its grid point cloud
    # get the blob id
    blob_name = blob_data['blobID']
    if (output_path / (blob_name+"_grid_point_cloud_fofc.xyzrgb")).exists() and not overwrite_pc:
        if verbose:
            logging.info("* Already processed blob "+ blob_name + ". Skipping to next blob. *")
        return pd.DataFrame.from_records([(True, 0)])
    elif blob_data["blobVolume"] == 0:
        # skip blobs with zero volume
        if verbose:
            logging.info("No volume was found for this blob " + blob_name + ". Skipping to next blob.")
        return pd.DataFrame.from_records([(False, 0)])
    if verbose:
        logging.info("\n* Start processing blob ID" + blob_name + "*")
    t1 = time.time()
    # get the blob center position xyz and the box bound size
    xyz = blob_data[['x','y','z']].values.astype(np.float64)
    #xyz_bound = np.asarray([float(blob_data['xyz_bound'])]*3)
    xyz_bound = np.asarray([float(i) for i in blob_data['xyz_bound'].split(",")])
    # print("GRID xyz_bound", xyz_bound)
    # shift to a position equals the <scale_boundingbox> times the bound size from the blob center and
    # create a 3d grid <scale_boundingbox> times the original bounding box size
    # the position xyz0 is the 0,0,0 point of the 3d grid
    scale_boundingbox = 1.2
    xyz0 = (xyz-scale_boundingbox*xyz_bound/2).round()
    grid_size = (xyz_bound*scale_boundingbox/grid_space).astype(int)
    if verbose:
        logging.info("Grid Info:\nCenter x,y,z = " + str(xyz) + "\nbounding box size lx,ly,lz = " + str(xyz_bound) +
          "\nBounding box scaling = " + str(scale_boundingbox) +
          "\nGrid size = " + str(grid_size))
        #
        logging.info("- Interpolate the density map values in the blob bounding box grid")
    try:
        # read the map and extract the grid
        map_grid_fofc = gemmi.read_ccp4_map(map_grid_file, setup=True).grid
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
            logging.info("- Create the blob bounding box grid in a point cloud file")
        # create a point cloud with all the grid points to store the original xyz positions and values
        point_cloud_blob_grid_fofc = ""
        for ijk, v in np.ndenumerate(fofc_3d_grid):
            # transform the grid indices to the real space of the cristal
            xyz_real = (np.array(ijk) * grid_space + xyz0)
            point_cloud_blob_grid_fofc += ' '.join(np.char.mod('%.1f', xyz_real)) + \
                                       (" " + str(fofc_3d_grid[ijk])) * 3 +"\n" #+" 0 "+str(fofc_3d_grid[l, j, k])+"\n" #+ " 0 0\n"
        del fofc_3d_grid #,twofofc_3d_grid
        # save in the xyzrgb format:
        # Each line contains [x, y, z, r, g, b], where r, g, b are in floats of range [0, 1]
        open(output_path / (blob_name+"_grid_point_cloud_fofc.xyzrgb"), "w").write(point_cloud_blob_grid_fofc)
        # read point cloud fofc and set the blob point cloud grid column to True
        blod_grid = o3d.io.read_point_cloud(
            (output_path / (blob_name + "_grid_point_cloud_fofc.xyzrgb")).as_posix())
        if verbose:
            logging.info("Number of points in the blob grid" + str(len(blod_grid.points)))
        del blod_grid
    except Exception as e:
        if verbose:
            logging.info("Error interpolating the blob density values and creating the grid point cloud. Blob ID:" +
                  blob_name + "\nError msg:" + str(e))
        return pd.DataFrame.from_records([(False, 0)])
    #
    d = time.time() - t1
    if verbose:
        logging.info("Processed blob" + blob_name + "in: %.2f s." % d)
    #
    return pd.DataFrame.from_records([(True, d)])


class EngineBlobGridPc(object):
    def __init__(self, blobs_list, refinement_data_path, output_path,
                 overwrite_pc, grid_space, num_processors):
        self.blobs_list = blobs_list
        self.refinement_data_path = refinement_data_path
        self.entryID = refinement_data_path.name
        self.output_path = output_path
        self.overwrite_pc = overwrite_pc
        self.grid_space = grid_space
        self.verbose = (num_processors == 1)
        # get the entry map here
        self.map_grid_file = self.retrieve_entry_density_map_path()
    def retrieve_entry_density_map_path(self):
        # check if a provided map or the fofc ccp4 map exists for the img creation
        # and allow using a different map (entryID.ccp4) instead of the fofc
        if (self.refinement_data_path / (self.entryID + ".ccp4")).exists():
            logging.info(
                "* Another entry map named " + (self.refinement_data_path / (self.entryID + ".ccp4")).name +
                " was provided and will be used in the blob's grid creation. *")
            return (self.refinement_data_path / (self.entryID + ".ccp4")).as_posix()
        elif (self.refinement_data_path / (self.entryID + "_fofc.ccp4")).exists():
            logging.info(
                "* The Fo-Fc entry map named " + (self.refinement_data_path / (self.entryID + "_fofc.ccp4")).name +
                " was created and will be used in the blob's grid creation. *")
            return (self.refinement_data_path / (self.entryID + "_fofc.ccp4")).as_posix()
        else:
            logging.info("Could not find the entry density map file.")
            sys.exit("Error retrieving the entry density map! Skipping blob's grid creation.")
    # call blobs creation
    def __call__(self, i):
        return create_blob_grid_pc(self.map_grid_file, self.blobs_list.iloc[i, :], self.output_path,
                                   self.overwrite_pc, self.grid_space, self.verbose)


def create_blobs_grid_point_cloud(refinement_data_path, entry_output_path, overwrite_pc=False,grid_space=0.2,
                                  num_processors=2):
    logging.info("- grid_space: " + str(grid_space))
    t0 = time.time()
    if num_processors > cpu_count():
        num_processors = max(cpu_count() - 1, 1)
        logging.info("Warning: The selected number of processors is greater than the available CPUs, "
              "setting it to the number of CPUs - 1 = " + str(num_processors))

    # check inputs and create missing directories
    refinement_data_path = Path(refinement_data_path)
    blobs_list_file = entry_output_path / "blobs_list.csv"

    if not refinement_data_path.exists() or not refinement_data_path.is_dir():
        logging.info("The provided entry refinement data folder do not exists.")
        sys.exit("The provided entry refinement data folder do not exists.")

    if not Path(blobs_list_file).is_file():
        logging.info("The blobs_list.csv file is not present in the provided refinement path.")
        sys.exit("The blobs_list.csv file is not present in the provided refinement path.")

    output_path = entry_output_path / "model_blobs" / "blobs_grid"
    if not output_path.exists() or not output_path.is_dir():
        output_path.mkdir(parents=True)

    # read csv with the list of blobs to extract their grids
    blobs_list = pd.read_csv(blobs_list_file)
    n = blobs_list.shape[0]

    blobs_list["point_cloud_grid"] = False

    # for each entry create its blobs grids
    # run the blob grid pc creation in multiprocessing
    # create the engine to create the blob pc passing the parameters
    engine_createBlobsGridPc = EngineBlobGridPc(blobs_list, refinement_data_path,
                                               output_path, overwrite_pc, grid_space, num_processors)
    blobs_grid_out = p_map(engine_createBlobsGridPc.__call__, range(n),
                           num_cpus=num_processors)
    # convert the result in a dataframe and merge with the ligands data
    blobs_grid_out = pd.concat(blobs_grid_out).reset_index(drop=True)
    blobs_list[["point_cloud_grid", "time_process"]] = blobs_grid_out
    # store the entry resolution and the used grid_space
    # blobs_list["resolution"] = engine_createBlobsGridPc.resolution
    # blobs_list["grid_space"] = engine_createBlobsGridPc.grid_space
    logging.info("Average time spent to process blobs: %.2f s." % blobs_list.time_process.mean())

    # print number of blob processing with error
    logging.info("\n")

    if sum(~blobs_list.point_cloud_grid) > 0:
        logging.info("A total of "+str(sum(~blobs_list.point_cloud_grid))+"/"+ str(n)+
              " blob entries raised an error and could not be entirely processed. The following entries will be "+
              "removed from the list: \n"+ str(blobs_list.loc[~blobs_list.point_cloud_grid, ["entryID","blobID"]]))
    else:
        logging.info("All blob's grids were successfully created!")
        logging.info("Total blob entries: " + str(n))

    blobs_list = blobs_list.loc[blobs_list.point_cloud_grid, :]

    # save processed blobs
    blobs_list.to_csv((output_path / "blobs_list_processed.csv").as_posix(), index=False)

    d = time.time() - t0
    logging.info("\nDONE! " + "Time elapsed: %.2f s." % d)
    return blobs_list.shape[0]


if __name__ == "__main__":
    # read the ligands csv with respective pdb entries, x, y, z and data
    overwrite_pc = False
    num_processors = 2
    grid_space = 0.5

    if len(sys.argv) >= 3:
        refinement_data_path = sys.argv[1]
        entry_output_path = sys.argv[2]
        if len(sys.argv) >= 4:
            overwrite_pc = (sys.argv[3].lower() == "true")
        if len(sys.argv) >= 5:
            num_processors = int(sys.argv[4])
        if len(sys.argv) >= 6:
            grid_space = float(sys.argv[5])
    else:
        sys.exit("Wrong number of arguments. One argument must be supplied in order to read an entry refinement "
                 "path and extract the grip pointclouds of the list of blobs: \n"
                 "  1. The path to the data folder where the entry refinement is located ('data/refinement/entryID'). "
                 "It must contain the table 'blobs_list.csv' file resulting from the parse_find_blobs_fofc.py script; "
                 "and the pdb and mtz files resulting from the refinement process with dimple;\n"
                 "  2. The path to the output data folder where the np3 ligand result is being stored for the current"
                 "entry ('data/np3_ligand_<DATE>/entryID');\n"
                 "  3. (optional) A boolean True or False indicating if the already processed blobs should be "
                 "overwritten (Default to False).\n"
                 "  4. (optional) The number of processors to use for multiprocessing parallelization (Default to 2).\n"
                 "  5. (optional) grid_space: The grid space size in angstroms to be used to create the point clouds "
                 "(Default to 0.5 A). It should be the same of the used model.\n"
                 )


    create_blobs_grid_point_cloud(refinement_data_path, entry_output_path, overwrite_pc, grid_space, num_processors)
