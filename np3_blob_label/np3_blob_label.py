from src_.refinement_dimple import refine_entries
from src_.find_blobs_fofc import find_blobs_parse_place_fake_atoms, search_blobs_parse_place_fake_atoms
from src_.mtz_to_blob_grid_pointcloud import create_blobs_grid_point_cloud
from src_.blob_grid_pointcloud_to_quantile_rank_scale import create_blobs_dataset_point_cloud_quantile_rank_scaled
from src_.predict_blob import predict_blobs
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys, os
import logging
from time import strftime, localtime, time

from config import get_config

# find all the blobs in an entry
def create_final_blobs_prediction_report(entries_list, np3_output_path):
    final_blobs_prediction_report = pd.DataFrame()

    for entryID in entries_list.entryID.unique():
        entry_blobs_report_path = np3_output_path / entryID / (entryID+"_blobs_list_prediction_report.csv")
        if entry_blobs_report_path.exists():
            entry_blobs_report = pd.read_csv(entry_blobs_report_path)
            final_blobs_prediction_report = pd.concat([final_blobs_prediction_report, entry_blobs_report])

    n_successful_blobs = final_blobs_prediction_report.shape[0]
    if n_successful_blobs > 0:
        final_blobs_prediction_report.to_csv(np3_output_path / "entries_final_blobs_prediction_report.csv", index=False)

    return n_successful_blobs

#########
# search for blobs in the entire map - unique entryID
######### find_all blobs_label
def np3_ligand_blob_label(db_path, entries_list_path, model_ckpt_path, grid_space, num_gpu, gpu_index, num_workers = 1,
                          sigma_cutoff=3, blob_min_volume=20, blob_min_score=10, blob_min_peak=0,
                          num_processors = 2, refinement_path = None,
                          overwrite_grid_pc = False, overwrite_blob_pc = False):
    # initialize output path to store the np3 results
    # pass this output path to the workflow steps
    # add output_name value, if not empty add an underscore
    if config.output_name != "":
        config.output_name = config.output_name + "_"
    np3_output_path = Path(db_path) / strftime("np3_ligand_" + config.output_name + "%Y%m%d_%Hh%Mm%Ss", localtime())
    if not np3_output_path.exists() or not np3_output_path.is_dir():
        np3_output_path.mkdir(parents=True)

    # setup logging
    # set the stdout logging level
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    # add logging file handler
    fh = logging.FileHandler(np3_output_path.as_posix() + '/np3_blob_label_output.log')
    fh.setLevel(logging.INFO)
    # set basic config to both handlers
    logging.basicConfig(
        format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
        datefmt='%m/%d %H:%M:%S',
        handlers=[ch, fh])
    # add the handlers to the logger
    logging.getLogger().addHandler(fh)
    # log the parameters:
    logging.info("Workflow: ")
    logging.info("- search_blobs: " + str(config.search_blobs))
    logging.info("Arguments: ")
    logging.info("- db_path " + str(db_path))
    logging.info("- entries_list_path " + str(entries_list_path))
    logging.info("- model_ckpt_path " + str(model_ckpt_path))
    logging.info("- grid_space " + str(grid_space))
    logging.info("- num_gpu " + str(num_gpu))
    logging.info("- gpu_index " + str(gpu_index))
    logging.info("- num_workers " + str(num_workers))
    logging.info("- sigma_cutoff " + str(sigma_cutoff))
    logging.info("- blob_min_volume " + str(blob_min_volume))
    logging.info("- blob_min_score " + str(blob_min_score))
    logging.info("- blob_min_peak " + str(blob_min_peak))
    logging.info("- num_processors " + str(num_processors))
    logging.info("- refinement_path " + str(refinement_path))
    # logging.info("- overwrite_grid_pc " + str(overwrite_grid_pc))
    # logging.info("- overwrite_blob_pc " + str(overwrite_blob_pc))
    logging.info("\n\n")

    # run the entries refinement if no refinement_path was provided or
    # if the refinement_entries_list.csv do not exists
    logging.info("\n")
    logging.info("**********************************")
    logging.info("* Step 1: Refinement with Dimple *")
    logging.info("**********************************\n")

    t0 = time()
    if refinement_path is None or not Path(refinement_path).exists():
        # if refinement_path does not exists, stop here
        if refinement_path is not None:
            logging.info("-> The provided refinement_path does not exists!! =/ Enable running Dimple to refine the "
                     "entries or correct the provided path.")
            sys.exit(1)
        refinement_path = refine_entries(db_path, entry_list_path=entries_list_path, num_processors=num_processors,
                       refinement_path=refinement_path)
        dt = time() - t0
        logging.info("  - Done in : %.2f s.!" % dt)
        refinement_path = Path(refinement_path)
        entries_list_path = Path(refinement_path) / "refinement_entries_list.csv"
    else:
        logging.info("  - Refinement result already provided. Skipping Step 1.")
        refinement_path = Path(refinement_path)
        entries_list_path = Path(entries_list_path)

    if not entries_list_path.exists() or not entries_list_path.is_file():
        logging.info("* ERROR: The provided entries_list_path does not exists -> "+entries_list_path.as_posix())
        logging.info("* ERROR: Please provide a valid path to your datasets metadata table and retry!")
        sys.exit(1)

    if not Path(model_ckpt_path).exists() or not Path(model_ckpt_path).is_file():
        logging.info("* ERROR: The provided model checkpoint file does not exists -> "+model_ckpt_path)
        logging.info("* ERROR: Please provide a valid path to a model checkpoint file that you want to use for "
                     "the blobs prediction and retry!")
        sys.exit(1)

    # read the list of refined entries
    entries_list = pd.read_csv(entries_list_path)
    if entries_list.columns.isin(['entryID', 'refinement', 'noHetatm']).sum() != 3:
        logging.info("* ERROR: The provided entries_list_path do not have all the *mandatory* columns: "
                     "'entryID', 'refinement' and 'noHetatm' -> " + entries_list_path.as_posix())
        logging.info("* ERROR: Please provide a valid path to your datasets metadata table with all the mandatory " +
                     "columns correctly created and retry.")
        sys.exit(1)
    # remove duplicated entries
    if entries_list.entryID.duplicated().any():
        logging.info("* Warning: There are duplicated entryID values in the entries_list metadata table. "
                     "Duplicated values will be removed.")
        entries_list = entries_list.loc[~entries_list.entryID.duplicated(), :]
    logging.info("** Number of refined entries: "+ str(entries_list.shape[0])+" **")
    success_processed_entries = 0
    entries_no_modeled_blob = []
    success_processed_blobs = 0
    total_n_blobs = 0
    # for each entry, find blobs, create the blobs image and run the model prediction
    for i in tqdm(range(entries_list.shape[0]), desc="Processed entries"):
        t0 = time()
        logging.info("\n\n")
        logging.info("******************" + "*" * (
                    len(entries_list.entryID[i]) + len(str(i)) + len(str(entries_list.shape[0]))) +
              "***********************")
        logging.info("******************" + "*" * (
                    len(entries_list.entryID[i]) + len(str(i)) + len(str(entries_list.shape[0]))) +
              "***********************")
        logging.info("** Processing entry "+ entries_list.entryID[i]+ " (" + str(i)+ "/"+ str(entries_list.shape[0])+
                     ") - Steps 2 to 5 **")
        logging.info("******************" + "*" * (
                    len(entries_list.entryID[i]) + len(str(i)) + len(str(entries_list.shape[0]))) +
              "***********************")
        logging.info("******************"+"*"*(len(entries_list.entryID[i])+len(str(i))+len(str(entries_list.shape[0])))+
              "***********************")
        logging.info("\n\n")
        entry_ref_path = refinement_path / entries_list.entryID[i]
        entry_output_path = np3_output_path / entries_list.entryID[i]
        # for entry in entry_list: run the rest of the pipeline:
        # 2. find blobs
        logging.info("************************************")
        logging.info("* Step 2: Find blobs by flood fill *")
        logging.info("************************************")
        logging.info("\n")
        t1 = time()
        n_blobs = find_blobs_parse_place_fake_atoms(entry_ref_path, entry_output_path, grid_space,
                                                    sigma_cutoff=sigma_cutoff, blob_min_volume=blob_min_volume,
                                                    blob_min_score=blob_min_score, blob_min_peak=blob_min_peak)
        total_n_blobs += n_blobs
        if n_blobs == 0:
            logging.info("\n\n")
            logging.info("- No blobs were found. Skipping entry "+ entries_list.entryID[i])
            entries_no_modeled_blob.append(entries_list.entryID[i])
            entry_output_path.rmdir()  # remove empty dir
            continue
        else:
            logging.info("\n\n")
            logging.info("- Total number of blobs found: "+ str(n_blobs))
            logging.info("- Elapsed time: %.2f s" % (time() - t1))
        logging.info("\n")
        logging.info("***********************************")
        logging.info("* Step 3: Create blobs grid image *")
        logging.info("***********************************\n")
        t1 = time()
        # 3. extract blobs image to a pc grid
        n_blobs = create_blobs_grid_point_cloud(entry_ref_path, entry_output_path, overwrite_pc=overwrite_grid_pc,
                                                grid_space=grid_space, num_processors=num_processors)
        if n_blobs == 0:
            logging.info("  - No blobs grid image could be created. Skipping entry "+ entries_list.entryID[i])
            entries_no_modeled_blob.append(entries_list.entryID[i])
            continue
        else:
            logging.info("  - Total of blobs' grid image created: " + str(n_blobs))
            logging.info("  - Elapsed time: %.2f s" % (time() - t1))
        logging.info("\n")
        logging.info("******************************************")
        logging.info("* Step 3: Create blobs point cloud image *")
        logging.info("******************************************\n")
        t1 = time()
        # 4. create blobs image in qrank scale in point clouds
        n_blobs = create_blobs_dataset_point_cloud_quantile_rank_scaled(refinement_path=entry_ref_path,
                                                                        entry_output_path=entry_output_path,
                                                              num_processors=num_processors,
                                                              overwrite_pc=overwrite_blob_pc)
        success_processed_blobs += n_blobs
        if n_blobs == 0:
            logging.info("  - No blobs point cloud image could be created. Skipping entry "+ entries_list.entryID[i])
            entries_no_modeled_blob.append(entries_list.entryID[i])
            continue
        else:
            logging.info("  - Total of blobs' point cloud image created: "+ str(n_blobs))
            logging.info("  - Elapsed time: %.2f s" % (time() - t1))
        logging.info("\n")
        logging.info("********************************************************************")
        logging.info("* Steps 4 and 5: Predict the blobs images and convert to CCP4 maps *")
        logging.info("********************************************************************\n")
        t1 = time()
        # 5. Predict the model classes in the blobs images and convert the pc result to CCP4 maps, one by class with all blobs
        # also create a pdb with fake atoms pointing to the blobs position
        # num_gpu max to 1. Com +2 a predicao pode vir em outra formatacao.. similar a um batch_size > 1
        pred_res = predict_blobs(entry_output_path=entry_output_path.as_posix(),
                      entry_refinement_path=entry_ref_path, model_ckpt_path=model_ckpt_path,
                      batch_size=1, num_gpu=num_gpu, gpu_index=gpu_index, num_workers=num_workers)
        if pred_res:
            # logging.info(pred_res)
            logging.info("  - Done!")
        else:
            logging.info("  - Error")
        logging.info("  - Elapsed time: %.2f s" % (time() - t1))
        dt = time() - t0
        logging.info("\n")
        logging.info("*******************"+"*"*len(entries_list.entryID[i])+"*****************")
        logging.info("*******************" + "*" * len(entries_list.entryID[i]) + "*****************")
        logging.info("** Finish for entry "+ entries_list.entryID[i]+          " in : %.2f s  **" % dt)
        logging.info("*******************" + "*" * len(entries_list.entryID[i]) + "*****************")
        logging.info("*******************"+"*"*len(entries_list.entryID[i])+"*****************\n")
        success_processed_entries += 1

    success_prediction_report = create_final_blobs_prediction_report(entries_list, np3_output_path)
    if success_prediction_report != success_processed_blobs:
        logging.info("-> ERROR matching the number of successfully processed blobs ("+str(success_processed_blobs)+
                     ") with the number of blobs present in the final report ("+str(success_prediction_report)+
                     ") - something went wrong in the counting.")
    logging.info("-> Number of successfully processed entries: " +
                 str(success_processed_entries) + "/" + str(entries_list.shape[0]))
    if len(entries_no_modeled_blob) > 0:
        logging.info("  -> Entries without any found blob: " +
                     str(entries_no_modeled_blob))
    logging.info("-> Number of successfully processed blobs from the total number of blobs found: " +
                 str(success_processed_blobs) + "/" + str(total_n_blobs))
    logging.info("\n-> See log file np3_blob_label_output.log")

#########
# search for blobs in specific positions and given blobIds
######### search_blobs_specific_pos_label
def np3_ligand_listed_blob_label(db_path, blobs_list_path, model_ckpt_path, grid_space, num_gpu, gpu_index, num_workers = 1,
                          sigma_cutoff=3.0, blob_min_volume=20, blob_min_score=10, blob_min_peak=0,
                          num_processors = 2, refinement_path = None,
                          overwrite_grid_pc = False, overwrite_blob_pc = False):

    blobs_list_path = Path(blobs_list_path)
    # initialize output path to store the np3 results
    # pass this output path to the workflow steps
    # add output_name value, if not empty add an underscore
    if config.output_name != "":
        config.output_name = config.output_name + "_"
    np3_output_path = Path(db_path) / strftime("np3_ligand_" + config.output_name + "%Y%m%d_%Hh%Mm%Ss", localtime())
    if not np3_output_path.exists() or not np3_output_path.is_dir():
        np3_output_path.mkdir(parents=True)

    # setup logging
    # set the stdout logging level
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    # add logging file handler
    fh = logging.FileHandler(np3_output_path.as_posix() + '/np3_blob_label_output.log')
    fh.setLevel(logging.INFO)
    # set basic config to both handlers
    logging.basicConfig(
        format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
        datefmt='%m/%d %H:%M:%S',
        handlers=[ch, fh])
    # add the handlers to the logger
    logging.getLogger().addHandler(fh)
    # log the parameters:
    logging.info("Workflow: ")
    logging.info("- search_blobs: " + str(config.search_blobs))
    logging.info("Arguments: ")
    logging.info("- db_path " + str(db_path))
    logging.info("- entries_list_path " + str(blobs_list_path))
    logging.info("- model_ckpt_path " + str(model_ckpt_path))
    logging.info("- grid_space " + str(grid_space))
    logging.info("- num_gpu " + str(num_gpu))
    logging.info("- gpu_index " + str(gpu_index))
    logging.info("- num_workers " + str(num_workers))
    logging.info("- sigma_cutoff " + str(sigma_cutoff))
    logging.info("- blob_min_volume " + str(blob_min_volume))
    logging.info("- blob_min_score " + str(blob_min_score))
    logging.info("- blob_min_peak " + str(blob_min_peak))
    logging.info("- num_processors " + str(num_processors))
    logging.info("- refinement_path " + str(refinement_path))
#    logging.info("- overwrite_grid_pc " + str(overwrite_grid_pc))
#    logging.info("- overwrite_blob_pc " + str(overwrite_blob_pc))
    logging.info("\n\n")

    # run the entries refinement if no refinement_path was provided or
    # if the refinement_entries_list.csv do not exists
    logging.info("\n")
    logging.info("**********************************")
    logging.info("* Step 1: Refinement with Dimple *")
    logging.info("**********************************\n")

    t0 = time()
    if refinement_path is None or not Path(refinement_path).exists():
        # if refinement_path does not exists, stop here
        if refinement_path is not None:
            logging.info("-> The provided refinement_path does not exists!! =/ Enable running Dimple to refine the "
                     "entries or correct the provided path.")
            sys.exit(1)
        refinement_path = refine_entries(db_path, entry_list_path=blobs_list_path, num_processors=num_processors,
                       refinement_path=refinement_path)
        dt = time() - t0
        logging.info("  - Done in : %.2f s.!" % dt)
        refinement_path = Path(refinement_path)
        entries_list_path = Path(refinement_path) / "refinement_entries_list.csv"
    else:
        logging.info("  - Refinement result already provided. Skipping Step 1.")
        refinement_path = Path(refinement_path)
        entries_list_path = blobs_list_path

    if not blobs_list_path.exists() or not blobs_list_path.is_file():
        logging.info("* ERROR: The provided blobs_list_path does not exists -> "+blobs_list_path.as_posix())
        logging.info("* ERROR: Please provide a valid path to your datasets metadata table with the blobs positions " +
                     "specified and retry!")
        sys.exit(1)

    if not entries_list_path.exists() or not entries_list_path.is_file():
        logging.info("* ERROR: The provided entries_list_path does not exists -> "+entries_list_path.as_posix())
        logging.info("* ERROR: Probable something went wrong in the refinement step, "
                     "look for error messages to fix them and retry.")
        sys.exit(1)

    if not Path(model_ckpt_path).exists() or not Path(model_ckpt_path).is_file():
        logging.info("* ERROR: The provided model checkpoint file does not exists -> "+model_ckpt_path)
        logging.info("* ERROR: Please provide a valid path to a model checkpoint file that you want to use for "
                     "the blobs prediction and retry!")
        sys.exit(1)

    # read the list of refined entries and of blobs to search
    entries_list = pd.read_csv(entries_list_path)
    if entries_list.columns.isin(['entryID', 'refinement', 'noHetatm']).sum() != 3:
        logging.info("* ERROR: The provided entries_list_path do not have all the *mandatory* columns: "
                     "'entryID', 'refinement' and 'noHetatm' -> " + entries_list_path.as_posix())
        logging.info("* ERROR: Probable something went wrong in the refinement step, "
                     "look for error messages to fix them and retry.")
        sys.exit(1)

    if entries_list_path == blobs_list_path:
        # refinement was provided, filter the unique entries
        entries_list = entries_list.loc[~entries_list.entryID.duplicated(), ["entryID", "refinement"]].reset_index(drop=True)
        logging.info("** Number of refined entries that were provided in the metadata table: " + str(entries_list.shape[0]) + " **")
    else:
        logging.info("** Number of successfully refined entries: " + str(entries_list.shape[0]) + " **")
    # read the blobs list and check if the mandatory columns are present
    blobs_list = pd.read_csv(blobs_list_path)
    if blobs_list.columns.isin(['entryID', 'blobID', 'refinement', 'noHetatm', 'x', 'y', 'z']).sum() != 7:
        logging.info("* ERROR: The provided blobs_list_path do not have all the *mandatory* columns: "
                     "'entryID', 'blobID', 'refinement', 'noHetatm', 'x', 'y' and 'z' -> " + blobs_list_path.as_posix())
        logging.info("* ERROR: Please provide a valid path to your datasets metadata table with all the mandatory " +
                     "columns correctly created and retry!")
        sys.exit(1)

    total_n_blobs = blobs_list.shape[0]
    # filter only the blobs that appear in the refined entries
    blobs_list = blobs_list.loc[blobs_list.entryID.isin(entries_list.entryID),:]

    success_processed_entries = 0
    entries_no_modeled_blob = []
    success_processed_blobs = 0
    # for each entry, find blobs, create the blobs image and run the model prediction
    for i in tqdm(range(entries_list.shape[0]), desc="Processed entries"):
        t0 = time()
        logging.info("\n\n")
        logging.info("******************" + "*" * (
                    len(entries_list.entryID[i]) + len(str(i)) + len(str(entries_list.shape[0]))) +
              "***********************")
        logging.info("******************" + "*" * (
                    len(entries_list.entryID[i]) + len(str(i)) + len(str(entries_list.shape[0]))) +
              "***********************")
        logging.info("** Processing entry "+ entries_list.entryID[i]+ " (" + str(i)+ "/"+ str(entries_list.shape[0])+
                     ") - Steps 2 to 5 **")
        logging.info("******************" + "*" * (
                    len(entries_list.entryID[i]) + len(str(i)) + len(str(entries_list.shape[0]))) +
              "***********************")
        logging.info("******************"+"*"*(len(entries_list.entryID[i])+len(str(i))+len(str(entries_list.shape[0])))+
              "***********************")
        logging.info("\n\n")
        entry_ref_path = refinement_path / entries_list.entryID[i]
        entry_output_path = np3_output_path / entries_list.entryID[i]
        # for entry in entry_list: run the rest of the pipeline:
        # 2. find blobs
        logging.info("*****************************************************************")
        logging.info("* Step 2: Search for blobs in the given positions by flood fill *")
        logging.info("*****************************************************************")
        logging.info("\n")
        t1 = time()
        # call search blobs
        n_blobs = search_blobs_parse_place_fake_atoms(blobs_list.loc[blobs_list.entryID.isin([entries_list.entryID[i]]),
                                                                     ['entryID', 'blobID', 'refinement', 'noHetatm',
                                                                      'x', 'y', 'z']].reset_index(drop=True),
                                                      entry_ref_path, entry_output_path, grid_space,
                                                      sigma_cutoff=sigma_cutoff)
        if n_blobs == 0:
            logging.info("\n\n")
            logging.info("- No blobs were found. Skipping entry "+ entries_list.entryID[i])
            entries_no_modeled_blob.append(entries_list.entryID[i])
            entry_output_path.rmdir()  # remove empty dir
            continue
        else:
            logging.info("\n\n")
            logging.info("- Total of blobs found: "+ str(n_blobs))
            logging.info("- Elapsed time: %.2f s" % (time() - t1))
        logging.info("\n")
        logging.info("***********************************")
        logging.info("* Step 3: Create blobs grid image *")
        logging.info("***********************************\n")
        t1 = time()
        # 3. extract blobs image to a pc grid
        n_blobs = create_blobs_grid_point_cloud(entry_ref_path, entry_output_path, overwrite_pc=overwrite_grid_pc,
                                                grid_space=grid_space, num_processors=num_processors)
        if n_blobs == 0:
            logging.info("  - No blobs grid image could be created. Skipping entry "+ entries_list.entryID[i])
            entries_no_modeled_blob.append(entries_list.entryID[i])
            continue
        else:
            logging.info("  - Total of blobs' grid image created: " + str(n_blobs))
            logging.info("  - Elapsed time: %.2f s" % (time() - t1))
        logging.info("\n")
        logging.info("******************************************")
        logging.info("* Step 3: Create blobs point cloud image *")
        logging.info("******************************************\n")
        t1 = time()
        # 4. create blobs image in qrank scale in point clouds
        n_blobs = create_blobs_dataset_point_cloud_quantile_rank_scaled(refinement_path=entry_ref_path,
                                                                        entry_output_path=entry_output_path,
                                                              num_processors=num_processors,
                                                              overwrite_pc=overwrite_blob_pc)
        success_processed_blobs += n_blobs
        if n_blobs == 0:
            logging.info("  - No blobs point cloud image could be created. Skipping entry "+ entries_list.entryID[i])
            entries_no_modeled_blob.append(entries_list.entryID[i])
            continue
        else:
            logging.info("  - Total of blobs' point cloud image created: "+ str(n_blobs))
            logging.info("  - Elapsed time: %.2f s" % (time() - t1))
        logging.info("\n")
        logging.info("********************************************************************")
        logging.info("* Steps 4 and 5: Predict the blobs images and convert to CCP4 maps *")
        logging.info("********************************************************************\n")
        t1 = time()
        # 5. Predict the model classes in the blobs images and convert the pc result to CCP4 maps, one by class with all blobs
        # also create a pdb with fake atoms pointing to the blobs position
        # num_gpu max to 1. Com +2 a predicao pode vir em outra formatacao.. similar a um batch_size > 1
        predict_blobs(entry_output_path=entry_output_path.as_posix(),
                      entry_refinement_path=entry_ref_path, model_ckpt_path=model_ckpt_path,
                      batch_size=1, num_gpu=num_gpu, gpu_index=gpu_index, num_workers=num_workers)
        logging.info("  - Done!")
        logging.info("  - Elapsed time: %.2f s" % (time() - t1))
        dt = time() - t0
        logging.info("\n")
        logging.info("*******************"+"*"*len(entries_list.entryID[i])+"*****************")
        logging.info("*******************" + "*" * len(entries_list.entryID[i]) + "*****************")
        logging.info("** Finish for entry "+ entries_list.entryID[i]+          " in : %.2f s  **" % dt)
        logging.info("*******************" + "*" * len(entries_list.entryID[i]) + "*****************")
        logging.info("*******************"+"*"*len(entries_list.entryID[i])+"*****************\n")
        success_processed_entries += 1

    success_prediction_report = create_final_blobs_prediction_report(entries_list, np3_output_path)
    if success_prediction_report != success_processed_blobs:
        logging.info("-> ERROR matching the number of successfully processed blobs (" + str(success_processed_blobs) +
                     ") with the number of blobs present in the final report (" + str(success_prediction_report) +
                     ") - something went wrong in the counting. :/")
    logging.info("-> Number of successfully processed entries: " +
                 str(success_processed_entries) + "/"+ str(entries_list.shape[0]))
    if len(entries_no_modeled_blob) > 0:
        logging.info("  -> Entries without any found blob: " +
                     str(entries_no_modeled_blob))
    logging.info("-> Number of successfully processed blobs: " +
                 str(success_processed_blobs) + "/" + str(total_n_blobs))
    logging.info("\n-> See log file np3_blob_label_output.log")


if __name__ == "__main__":
    config = get_config()
    if config.search_blobs == 'all':
        np3_ligand_blob_label(config.data_folder, config.entries_list_path, config.model_ckpt_path, config.grid_space,
                              config.num_gpu, config.gpu_index,
                              num_workers=config.num_workers, sigma_cutoff=config.sigma_cutoff,
                              blob_min_volume=config.blob_min_volume,
                              blob_min_score=config.blob_min_score, blob_min_peak=config.blob_min_peak,
                              num_processors=config.num_processors, refinement_path=config.refinement_path,
                              overwrite_grid_pc=config.overwrite_grid_pc, overwrite_blob_pc=config.overwrite_blob_pc)
    elif config.search_blobs == 'list':
        np3_ligand_listed_blob_label(config.data_folder, config.entries_list_path, config.model_ckpt_path, config.grid_space,
                              config.num_gpu, config.gpu_index,
                              num_workers=config.num_workers, sigma_cutoff=config.sigma_cutoff,
                              blob_min_volume=config.blob_min_volume,
                              blob_min_score=config.blob_min_score, blob_min_peak=config.blob_min_peak,
                              num_processors=config.num_processors, refinement_path=config.refinement_path,
                              overwrite_grid_pc=config.overwrite_grid_pc, overwrite_blob_pc=config.overwrite_blob_pc)

