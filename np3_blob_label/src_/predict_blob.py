# main training and testing pipeline, call the pytorch lightning module
import logging

# Torch packages
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
try:
    from pytorch_lightning.core import LightningModule
    from pytorch_lightning import Trainer
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install open3d pytorch_lightning`."
    )

from pytorch_lightning import loggers as pl_loggers

import numpy as np
import pandas as pd
import MinkowskiEngine as ME

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../np3_DL_segmentation")
from np3_DL_segmentation.src.utils import mkdir_p,load_state_with_same_shape
from np3_DL_segmentation.src.load_models import load_model
from np3_DL_segmentation.src.train_pytorchlightning import MinkowskiSegmentationModule
from src_.blob_pc_data_loader import *

from argparse import Namespace
import open3d as o3d
import gemmi
from tqdm import tqdm
from pathlib import Path

# Change dataloader multiprocess start method to anything not fork
# torch.multiprocessing.set_start_method('spawn')

# matplotlib palletes
colors_SP = ['0.011764705882352941,0.9882352941176471,0.0784313725490196', '0.60,0.60,0.60', '0,1,1', '1,1,0', '0.65490196,0.63529412,0.51764706', '0.81568627,0.78431373,0.55686275', '0.94901961, 0.51764706, 0.50980392',
             '1,0.53333333,0.06666667', '0.57647059,0.76862745,0.49019608','0.32941176,0.34117647,0.48627451','0.8627451,0.49803922,0.60784314', '0.58823529,0.67843137,0.78431373']

#colors_set1 = ['0.97,0.51,0.75', '0.60,0.60,0.60', '0.65,0.34,0.16', '1.00,1.00,0.20', '1.00,0.50,0.00',
#               '0.60,0.31,0.64', '0.30,0.69,0.29', '0.22,0.49,0.72', '0.89,0.10,0.11']
colors_tab21 = ['0.12,0.47,0.71', '0.60,0.60,0.60', '0.68,0.78,0.91', '1.00,0.50,0.05', '1.00,0.73,0.47',
                '0.17,0.63,0.17',
                '0.60,0.87,0.54', '0.84,0.15,0.16', '1.00,0.60,0.59', '0.58,0.40,0.74', '0.77,0.69,0.84',
                '0.55,0.34,0.29', '0.77,0.61,0.58', '0.89,0.47,0.76', '0.97,0.71,0.82', '0.50,0.50,0.50',
                '0.78,0.78,0.78', '0.74,0.74,0.13', '0.86,0.86,0.55', '0.09,0.75,0.81', '0.62,0.85,0.90']
colors_set1 = colors_SP


def save_blobs_predictions_singleMap(blobs_preds_coords, blobs_dataset, blobs_img_path, pc_type, entry_map_path,
                                     class_names, entry_id):
    """Save predictions results for each blob in the voxelized coords scale,
    save original coords and the predicted labels in point cloud and then convert to ccp4 map. Creates one map by class
    and one map with the ligand space"""
    # read entry map to extract the unit cell
    # get from fofc
    entry_map = gemmi.read_ccp4_map(entry_map_path, setup=True)
    entry_map.grid.fill(np.NaN)
    entry_map.update_ccp4_header(mode=0, update_stats=True)
    # create one grid by class_name
    grid_classes = [{'class_name': class_label, 'class_grid': entry_map.grid.clone(), 'class_idx': i, 'present': False}
                    for i, class_label in enumerate(class_names)]
    # store the classes frequency in each blob entry
    class_names_colsize = class_names+"_size"
    blobs_dataset[class_names_colsize] = 0.0
    # for each blob save the prediction in a pc and in the ccp4 map
    # for the map with the ligand space, filter only the classes that are not background (different from 0)
    # also create one map for each class separated
    for i in tqdm(range(len(blobs_preds_coords)), desc="Creating point cloud prediction images"):
        # extract prediction information and corresponding coordinates
        preds = blobs_preds_coords[i]['preds']
        coords = blobs_preds_coords[i]['coords']*blobs_dataset.grid_space[0]
        blob_id = blobs_dataset.loc[blobs_preds_coords[i]['batch_idx'], "blobID"]
        # save prediction to pc
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords[:, 1:])
        pcd.colors = o3d.utility.Vector3dVector(np.asarray([[v,v,v] for v in preds]))
        #o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(blobs_img_path + '/' + blob_id + '/' + blob_id +
                                 '_point_cloud_'+pc_type+'_predicted.xyzrgb', pcd)
        del pcd
        # save prediction to ccp4 map
        # set predict value in the coords position, coords index 0 is the batch idx
        # set a fixed value of 6 around each point and of 9 in the given point
        for j in range(len(preds)):
            #coords[j] = coords[j]*grid_space
            # ligand space do not set background points
            if preds[j] != 0:
                entry_map.grid.set_points_around(gemmi.Position(coords[j][1],
                                                                coords[j][2],
                                                                coords[j][3]),
                                                 radius=blobs_dataset.grid_space[0]*3/4,
                                                 value=6.0)
                entry_map.grid.set_value(coords[j][1],
                                         coords[j][2],
                                         coords[j][3], 9.0)
            # set point in the respective class grid
            grid_classes[preds[j]]['class_grid'].set_points_around(gemmi.Position(coords[j][1],
                                                                coords[j][2],
                                                                coords[j][3]),
                                                 radius=blobs_dataset.grid_space[0]*3/4,
                                                 value=6.0)
            grid_classes[preds[j]]['class_grid'].set_value(coords[j][1],
                                         coords[j][2],
                                         coords[j][3], 9.0)
            grid_classes[preds[j]]['present'] = True
        # store classes frequency in each blob
        classes_idx, classes_counts = preds.unique(return_counts=True)
        blobs_dataset.loc[blobs_preds_coords[i]['batch_idx'], class_names_colsize[classes_idx]] = classes_counts.numpy()

    # store the output map file names
    out_map_names = []
    # save the map with the ligand space
    # symmetrize maximum
    #entry_map.setup(np.NaN)
    entry_map.grid.symmetrize_max()
    # The setup function has two arguments. The first one is a value to be used for unknown values.
    # It is used only when the input file does not cover a complete asymmetric unit.
    # (If you used CCP4 program MAPMASK – it is keyword PAD there).
    # When you call a read function with setup=True, this argument is NaN for maps and -1 for masks.
    #entry_map.setup(np.NaN)
    entry_map.update_ccp4_header(mode=0, update_stats=True)
    #entry_map.grid.normalize()
    entry_map.write_ccp4_map(blobs_img_path + '/../' + entry_id+'_ligand_region.ccp4')
    out_map_names.append(blobs_img_path + '/../' + entry_id+'_ligand_region.ccp4')
    # save the map for each class
    classes_present = []
    for i in tqdm(range(len(class_names)), desc="Saving the CCP4 maps by predicted class"):
        # only save the not empty maps
        if grid_classes[i]['present']:
            entry_map.grid = grid_classes[i]['class_grid']
            #entry_map.setup(np.NaN)
            entry_map.grid.symmetrize_max()
            entry_map.update_ccp4_header(mode=0, update_stats=True)
            entry_map.write_ccp4_map(blobs_img_path + '/../' + entry_id + '_' +class_names[i]+'_class.ccp4')
            out_map_names.append(blobs_img_path + '/../' + entry_id + '_' +class_names[i]+'_class.ccp4')
            classes_present.append(i)
    # return the save maps
    return out_map_names, blobs_dataset[["blobID"]+list(class_names_colsize)], classes_present



def predict_blobs(entry_output_path, entry_refinement_path, model_ckpt_path, batch_size,
                  num_gpu, gpu_index, num_workers):
    entry_refinement_path = Path(entry_refinement_path)
    entry_output_path = Path(entry_output_path).resolve()
    entry_id = entry_output_path.name
    blobs_img_path = entry_output_path / 'model_blobs' / 'blobs_img'
    blobs_list_path = blobs_img_path / "blobs_list_processed_qRank_scale.csv"
    # config = get_config()
    # check if the num of gpus is valid
    if num_gpu > 0 and not torch.cuda.is_available():
        raise Exception("No GPU found")


    if num_gpu > 0 and gpu_index >= torch.cuda.device_count():
        raise Exception("Wrong GPU index ("+str(gpu_index)+"). The number of devices is "+
                        str(torch.cuda.device_count())+".")
    elif num_gpu == 0:
        # set fixed this parameters if is_cuda is False
        gpu_index = 0
        num_gpu = 0

    # set the stdout logging level
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    # add logging file handler
    # create file handler to log the training progress
    mkdir_p((entry_output_path / 'model_blobs' / 'prediction_logging').as_posix())
    fh = logging.FileHandler((entry_output_path / 'model_blobs' / 'prediction_logging').as_posix() + '/job_info.log')
    fh.setLevel(logging.INFO)
    # set basic config to both handlers
    logging.basicConfig(
        format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
        datefmt='%m/%d %H:%M:%S',
        handlers=[ch, fh])
    # add the handlers to the logger
    logging.getLogger().addHandler(fh)
    # create a logger for the model checkpoint and tensorboard info
    # logger as None - prevent saving unused info
    tb_logger = None  #pl_loggers.TensorBoardLogger(blobs_img_path.as_posix()+'/prediction_logging', name='model', default_hp_metric=False)

    logging.info('===> Configurations')
    dconfig = {'blobs_list_path': blobs_list_path.as_posix(),
               'blobs_img_path': blobs_img_path.as_posix(),
               'model_ckpt_path': model_ckpt_path,
               'num_gpu': num_gpu, 'gpu_index': gpu_index}
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    # get class names from the model.cpkt
    logging.info('===> Loading model checkpoint: ' + model_ckpt_path)
    # configure map_location properly, setting the main device when using gpu
    # map_location = ({'cuda:%d' % config.gpu_index: 'cuda:%d' % gpu} if gpu is not None else device)
    map_location = 'cpu'
    state = torch.load(model_ckpt_path, map_location=map_location)
    if 'state_dict' not in state.keys():
        state = {'state_dict': state}

    # remove model. prefix from the parameters names
    state['state_dict'] = {(k.partition('model.')[2] if k.startswith('model.') else k): state['state_dict'][k]
                           for k in state['state_dict'].keys() if
                           k not in ['criterion.weight', 'model.criterion.weight', 'criterion.cross_entropy.weight',
                                     'model.criterion.cross_entropy.weight']}
    # select fields to be printed - reduce output message 'model', 'pc_type'
    for k in ['model', 'pc_type']:
        logging.info('    {}: {}'.format(k, state['hyper_parameters'][k]))

    # extract class names and model name from hyper parameters, attribute it to the config var
    class_names = state['hyper_parameters']['class_names']
    model_name = state['hyper_parameters']['model']
    img_type = state['hyper_parameters']['pc_type']
    config = Namespace(**state['hyper_parameters'])
    # remove class weights for inference
    config.loss_weights = [1.0]

    ######
    logging.info('===> Building model')
    ######
    # use model name to select the ME UNet model
    net = load_model(model_name)
    # initialize model
    model = net(in_channels=(3 if "qRankMask_5_" in img_type else 1), out_channels=len(class_names), config=config, D=3)
    if num_gpu > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    # Load weights to the model
    model.load_state_dict(state['state_dict'])

    # create the blobs data loader
    blobs_data = blobs_dataloader(blobs_list_path.as_posix(), blobs_img_path.as_posix(), img_type, batch_size,
                                  num_workers)
    if not blobs_data:
        # if no blobs data, remove file handler, close it and return None
        logging.getLogger().removeHandler(fh)
        fh.close()
        return None

    # create the pytorch lightning module
    pl_module = MinkowskiSegmentationModule(config, model, tb_logger)
    ####
    # initialize trainer
    trainer = Trainer(max_epochs=1,
                      gpus=num_gpu,
                      accelerator="ddp",
                      log_every_n_steps=1,
                      flush_logs_every_n_steps=1,
                      logger=tb_logger,
                      accumulate_grad_batches=1,
                      stochastic_weight_avg=False)
    # predict with the blobs dataloader
    blobs_prediction = trainer.predict(pl_module, blobs_data)
    logging.info("")
    logging.info("Done predicting the blobs images!")
    # store prediction in a point cloud
    # and also save to a map, with all predictions together (ligand space)
    # and with one map by class with all blobs
    entry_map_path = entry_refinement_path / (entry_refinement_path.name+"_fofc.ccp4")
    output_maps_names, blobs_data_preds, classes_present = save_blobs_predictions_singleMap(blobs_prediction,
                                                                    blobs_data.dataset.blobs_retrieve,
                                                                    blobs_img_path.as_posix(), img_type,
                                                                    entry_map_path.as_posix(), class_names,
                                                                    entry_id)
    logging.info("")
    logging.info("Done converting the images predictions to CCP4 maps!")
    # create coot.py file to automatically open all maps together with the pdb and mtz file
    # read each map present in the output_maps_names, set different color to each one and contour level equals 2
    # similar to blob1-coot.py file, coping it as starting point
    # https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/web/docs/coot.html#set_002dlast_002dmap_002dcontour_002dlevel
    entry_view_script_path = entry_refinement_path / "entry-view-coot.py"
    coot_file = entry_view_script_path.read_text() + "\n\n"
    # read pdb with blobs
    coot_file = coot_file + "molecule_blobs = read_pdb('"+\
                (entry_output_path / entry_map_path.name.replace("_fofc.ccp4", "_blobs.pdb")).as_posix() +\
                "')\n"+\
                "set_rotation_centre("+str(blobs_data.dataset.blobs_retrieve.x[0])+", "+\
                str(blobs_data.dataset.blobs_retrieve.y[0])+", "+str(blobs_data.dataset.blobs_retrieve.z[0])+")\n"+\
                "set_show_symmetry_master(1)\n\n"

    # select color pallet according to the number of output maps]
    #print("CLASSES PRESENT = ",classes_present)
    if len(class_names)+1 <= len(colors_set1):
        color_maps = [colors_set1[0]]+[colors_set1[class_i+1] for class_i in classes_present]
    else:
        color_maps = [colors_tab21[0]]+[colors_tab21[class_i+1] for class_i in classes_present]

    #print("COLOR MAPS = ",color_maps)
    #print("OUTPUT MAPS NAMES = ",output_maps_names)
    # add the class maps reading in the coot script and save it
    for i, output_map in enumerate(output_maps_names):
        # start coot with the map of the ligand region with the view turned off - ease visualization of the predicted classes
        if i == 0:
            coot_file = coot_file + "imol_map_blob = handle_read_ccp4_map('" + output_map + "', 1)\n"
            coot_file = coot_file + "set_map_displayed(imol_map_blob, 0)\n"
        else:
            coot_file = coot_file + "handle_read_ccp4_map('"+output_map+ "', 1)\n"
        coot_file = coot_file + "set_last_map_contour_level_by_sigma(2)\n"
        coot_file = coot_file + "set_last_map_color("+color_maps[i]+")\n"

    blobs_view_path = entry_output_path / "prediction-blobs-view-coot.py"
    blobs_view_path.write_text(coot_file)
    logging.info("\n")
    logging.info("To see the results in coot run:")
    logging.info(">> coot --script "+ blobs_view_path.as_posix())
    # also save a report with the classes predicted for each blob
    blobs_data_preds_path = entry_output_path / (entry_id + "_blobs_list_prediction_report.csv")
    blobs_data = pd.read_csv(blobs_list_path)
    blobs_data_preds = blobs_data.merge(blobs_data_preds)
    # remove not used columns and the time column that came from the creation of the blob's final imgs in pcd
    blobs_data_preds.drop(["point_cloud_grid", "point_cloud", "missing_blob", 'time_process'], axis=1, inplace=True)
    blobs_data_preds.to_csv(blobs_data_preds_path, index=False)
    # close handler
    logging.getLogger().removeHandler(fh)
    fh.close()
    return 1



if __name__ == '__main__':
    batch_size = 1  # fixed to ease code, could be increased for high throughput performance
    # removed num_gpu from the arguments list similar to batch_size = 1, only use gpu_index ->
    #   if False use CPU, else use the given GPU index
    if len(sys.argv) >= 6:
        entry_output_path = sys.argv[1]
        entry_refinement_path = sys.argv[2]
        model_ckpt_path = sys.argv[3]
        # num_gpu = int(sys.argv[4])
        # gpu_index = int(sys.argv[5])
        gpu_index = sys.argv[4].lower()
        if gpu_index == "false":
            # CPU processing
            gpu_index = 0
            num_gpu = 0
        else:
            gpu_index = int(sys.argv[4])
            num_gpu = 1
        num_workers = int(sys.argv[5])
    else:
        sys.exit("Wrong number of arguments. Five arguments must be supplied in order to predict the classes of each "
                 "blob point cloud image using a provided model segmentation checkpoint. "
                 "Then convert the predicted point cloud images to a CCP4 map and also create one map by class with "
                 "all the blobs results. "
                 "Finally a coot scripting file is created to easy the visualization of the result with coot. "
                 "Arguments:\n"
                 "  1. blobs_img_path: The path to the folder named 'blobs_img' located in the 'model_blobs' folder of "
                 "the entry refinement directory. "
                 "The output point clouds prediction will be stored in each blob subfolder inside the 'blobs_img' "
                 "folder. And the output CCP4 maps with the predictions will be stored in the blobs_img_path "
                 "('blobs_img' folder);\n"
                 "  2. entry_refinement_path: The path to the data folder where the entry refinement is located "
                 "('data/refinement/entryID'). "
                 "It must contain the entry CCP4 map resulting from the blobs image creation and the coot script named "
                 "'entry-view-coot.py'. A coot script named 'prediction-blobs-view-coot.py' will be created in the "
                 "'model_blobs' subfolder to help visualizing the results;\n"
                 "  3. model_ckpt_path: The path to the segmentation model checkpoint file to be used in the "
                 "predictions for inferring the classes of each point present in the blobs point cloud image;\n"
                 # "  X. batch_size: a number defining the size of the batch size to be used in the model prediction, "
                 # "the number of images to be processed in each step;\n"
                 # "  X. num_gpu: a number defining the number of GPUs to be used in the model prediction;\n"
                 "  4. gpu_index: 'False' for CPU process or a integer number defining the GPU index to be used in "
                 "the segmentation process;\n"
                 "  5. num_workers: a number defining the number of workers to be used in the segmentation "
                 "model prediction;\n"
                 )
    predict_blobs(entry_output_path, entry_refinement_path, model_ckpt_path, batch_size,
                  num_gpu, gpu_index, num_workers)

