import argparse
from time import ctime


def str2bool(v):
    return v.lower() in ('true', '1', 't')


def str2list(l):
    return [int(i) for i in l.split(',')]


def str2flist(l):
    return [float(i) for i in l.split(',')]


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


arg_lists = []
parser = argparse.ArgumentParser(description="Runs the NP3 blob label pipeline with three mandatory arguments. "
                                             "This pipeline consists of 5 major steps: 1. refinement, 2. find blobs, "
                                             "3. create blobs 3D point clouds, 4. predict the blobs pc, and "
                                             "5. convert the predictions to CCP4 maps. \n"
                                             "The inputs are a metadata table defining the entries, "
                                             "the entries pdb and mtz files organized in folders or previous refinement results, "
                                             "the model to be used in the predictions and the find blobs parameters. "
                                             "The outputs are one CCP4 map by predicted class, a pdb file of the entry "
                                             "with fake atoms placed in the center position of the blobs, "
                                             "and a Coot python scripting file that ease the visualization of the "
                                             "results together with the input mtz and pdb files.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General
np3_blobs_arg = add_argument_group('Blob Search Mode')
np3_blobs_arg.add_argument('--search_blobs', type=str, default='all', choices=['all', 'list'],
                           help="Select the workflow that should be executed: "
                                "'all' will run the find blobs script (step 2) to search for all blobs that fulfill "
                                "the parameters criteria or "
                                "'list' will run step 2 to search for the blobs listed in specific locations in the "
                                "entries_list_path table \n")
np3_blobs_arg.add_argument('--parallel_cores', type=int, default=2,
                           help="The number of processors to use for multiprocessing parallelization."
                                " If 1, the parallelization is disabled and more verbose messages are emitted."
                                "The parallelization is applied in the refinement (step 1) and in the call to the point clouds "
                                "creation and prediction (steps 3 and 4). More memory is required for more cores.")


# refinement - dimple
data_arg = add_argument_group('Data In/Out')
data_arg.add_argument('--output_name', type=str, default="",
                           help="Used to name the output directory in following format: np3_blob_label_<output_name>_<DATE>. "
                                "Where DATE is the current date time and output_name is the given string or empty.\n")
data_arg.add_argument('--output_path', type=str, default=".",
                           help="The output directory path where the NP3 blob label pipeline output results will be stored. "
                                "The results will be stored to "
                                "a subfolder named with the output_name as np3_blob_label_<output_name>_<DATE>.\n")
data_arg.add_argument('--entries_list_path', type=str, required=True,
                           help="The path to the CSV metadata table defining the ligands dataset that will be processed (it must be comma separated). "
                                "It must contain one dataset entry by row and depending on the 'search_blobs' "
                                "parameter value it must contain the following mandatory columns:\n"
                                "- Three mandatory columns when 'search_blobs' is 'all': \n    "
                                "- 'entryID' with a unique entry name for each dataset (will be used to retrieve the "
                                "pdb and mtz data from the data_folder);\n"
                                "    - 'refinement' with 0 or 1 values defining if the given dataset should be refined "
                                "with Dimple (1) or not (0). If 0, the entry's pdb and mtz files present in the "
                                "data_folder will be copied to the entry refinement output directory; and\n"
                                "    - 'noHetatm' with 0 or 1 values defining if the hetero atoms present in the "
                                "entries .pdb files should be removed for the refinement (1 - it can remove "
                                "already modeled ligands) or not (0 - it will keep all ligands, useful to keep waters, "
                                "clusters and protein co-factors);\n"
                                "- When 'search_blobs' is 'list' the following four columns are *also* required (seven "
                                "mandatory columns in total):\n"
                                "    - 'blobID' with a unique identifier for each blob location;\n"
                                "    - 'x','y' and 'z' columns defining the center position of the blobs;\n")
data_arg.add_argument('--data_folder', type=str, default=None,
                           help="The PDB data folder path, if refinement_path is not informed, this data folder must have "
                                "the 'pdb' and the 'mtz' directories, containing the entries files named as "
                                "<entryID>.pdb and <entryID>.mtz, respectively. "
                                "Where the entryID is defined in the entries_list_path table. "
                                "A folder named 'refinement_<DATE>' will be created inside it to store the dimple results "
                                "(when applicable), with one subfolder for each refined entry present in the entry_list_table.\n")
data_arg.add_argument('--refinement_path', type=str, default=None,
                     help="The path to a previous refinement output directory containing one folder for each entry "
                          "refinement result, named with the respective <entryID> as defined in the entries_list_path table. "
                          "Used to continue from a previous result. "
                          "If it is 'None' a new refinement output directory is created. "
                          "A CCP4 map with the difference electron density map will be created inside the refinement "
                          "folder of each entry, named as <entryID>_fofc.ccp4. "
                          "If this file already exists it will not be overwritten.")

# find blobs
find_b_arg = add_argument_group('Find blobs')
find_b_arg.add_argument('--sigma_cutoff', type=float, default=3.0,
                        help="A numeric defining the sigma cutoff to be used to search for blobs in "
                             "the difference electron density map. Values closer to 3σ are recommended. "
                             "Values closer to 2σ may retrieve low quality blobs (which could have a fragmented density) "
                             "and values closer to 3σ may retrieve only high quality blobs."
                             "Values smaller than 1.5σ are *not recommended* due to memory constrains. For values smaller "
                             "or equal than 2.5σ, more than 32Gb of memory RAM is required.")
find_b_arg.add_argument('--blob_min_volume', type=float, default=24.0,
                        help="A numeric defining the minimum volume that a blob must have to be considered. Only used when search_blobs is 'all'. "
                             "Default to 24 A^3, what is equivalent to the volume of a water. "
                             "Smaller values will allow retrieving smaller blobs, the opposite is also true.")
find_b_arg.add_argument('--blob_min_score', type=float, default=0,
                        help="A numeric defining the minimum score that a blob must have to be considered. Only used when search_blobs is 'all'. "
                             "The score of a blob is equal   to the sum of the difference electron density values of "
                             "its points.")
find_b_arg.add_argument('--blob_min_peak', type=float, default=0,
                        help="A numeric defining the minimum intensity that the peak (most intense point) of a blob "
                             "must have for the blob to be considered. Only used when search_blobs is 'all'.")

# predict classes - segmentation of the blobs images
blob_segm_arg = add_argument_group('Prediction Model')
blob_segm_arg.add_argument('--model_ckpt_path', type=str, required=True,
                           help="The path to the model checkpoint file to be used in the predictions for "
                                "inferring the classes of each point present in the blobs point cloud - "
                                "segmentation result.\n")
blob_segm_arg.add_argument('--grid_space', type=float, default=0.5,
                          help="The grid space size in angstroms to be used to create the blobs "
                               "point cloud images. It must be the same of the one used in the model training. "
                               "The models provided in this repository use the default grid_space value equal 0.5.")
blob_segm_arg.add_argument('--gpu_index', type=int, default=None,
                           help="Do not inform any value (None) for CPU process (default) or an integer number defining the GPU index to be "
                                "used in the segmentation model prediction. Only one device is used in any case.")
blob_segm_arg.add_argument('--num_workers', type=int, default=1,
                           help="A numeric defining the number of workers to be used in the model prediction")



def get_config():
    config = parser.parse_args()

    # set the number of devices for CPU or GPU as 1 or set the provided gpu_index
    if config.gpu_index is not None:
        config.num_devices = [config.gpu_index]
    else:
        config.num_devices = 1

    # not used parameters, could be used in the future to skip steps
    config.overwrite_grid_pc = False  # false to allow using other <entryID>_fofc.ccp4 map file
    config.overwrite_blob_pc = True

    # set number of processor
    config.num_processors = config.parallel_cores

    return config  # np3 blob prediciton settings
