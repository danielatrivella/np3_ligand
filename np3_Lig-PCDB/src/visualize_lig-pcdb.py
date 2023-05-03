import open3d as o3d
import pandas as pd
import numpy as np
import sys, os


#elements_color_AtomSymbolGroups
elements_color_AtomSymbolGroups = {'0': np.array([0, 0, 0]), '1': np.array([144, 144, 144]) / 255,
                          '2': np.array([255, 13, 13]) / 255, '3': np.array([48, 80, 248]) / 255,
                           '4': np.array([255,128,0]) / 255, '5': np.array([31, 240, 31]) / 255}
elements_color_AtomSymbol = {'0': np.array([0, 0, 0]), '1': np.array([144, 144, 144]) / 255,
                          '2': np.array([255, 13, 13]) / 255, '3': np.array([48, 80, 248]) / 255,
                           '4': np.array([255,128,0]) / 255, '5': np.array([255, 255, 48]) / 255,
                            '6': np.array([255, 161, 0]) / 255, '7': np.array([166, 41, 41]) / 255, 
                            '8': np.array([31, 240, 31]) / 255, '9': np.array([144, 224, 80]) / 255, 
                            '10': np.array([148, 0, 148]) / 255}
#elements_color_ABC347CA56
elements_color_ABC347CA56 = {'0': np.array([0, 0, 0]), '1': np.array([0, 1, 1]),
                              '2': np.array([1, 1, 0]), '3': np.array([167, 162, 132]) / 255,
                          '4': np.array([208,200,142]) / 255, '5': np.array([242,132,130]) / 255,
                          '6': np.array([255, 136, 17]) / 255, '7': np.array([147, 196, 125]) / 255,
                          '8': np.array([84,87,124]) / 255, '9': np.array([220, 127, 155]) / 255,
                          '10': np.array([150, 173, 200]) / 255, '11': np.array([167, 194, 193]) / 255,
                          '12': np.array([183, 214, 186]) / 255, '13': np.array([215, 255, 171]) / 255,
                          '14': np.array([234, 255, 140]) / 255, '15': np.array([252, 255, 108]) / 255,
                          '16': np.array([242,132,130]) / 255, '17': np.array([163, 113, 91]) / 255,
                          '18': np.array([109, 69, 76]) / 255, '19': np.array([122, 86, 92]) / 255,
                          '20': np.array([230,145,56]) / 255, '21': np.array([0,255,105]) / 255,
                          '22': np.array([0, 1, 1]), '23': np.array([0, 1, 0]),
                          '24': np.array([0, 0.5, 0])}
elements_color_set = elements_color_ABC347CA56

def read_pc_labels(pcfile_labels, class_mapping, num_labels):
    if os.path.exists(pcfile_labels):
        labels = pd.read_table(pcfile_labels, header=None, delimiter=" ")[0].values.astype(np.int32)
        solvent_mask = (labels == -1)
        # map labels using the given mapping table where is not solvent label
        if class_mapping is not None:
            labels[~solvent_mask] = class_mapping[labels[~solvent_mask]]
            labels[solvent_mask] = class_mapping[-1]  # last target is always background
        else:
            labels[solvent_mask] = num_labels - 1 # last target is always background
        #print(np.unique(labels))
        return labels
    else:
        return np.asarray([])

# read_point_clouds_lig(pc_dir, '4xpq_FUL_A_1003', num_labels, class_mapping=class_mapping)
def read_point_clouds_lig(ligID, pc_dir, pred_directory, num_labels,
                          pc_types=["qRankMask","qRank0.5", "qRank0.75", "qRank0.95"],
                          class_mapping=None):
    entry = ligID.split('_')[0]
    lig_views = []
    # read all pc for the selected image types and translate them in the x-axis
    lig_pcds = [o3d.io.read_point_cloud(pc_dir + '/' + entry + '/' + ligID + '_lig_point_cloud_fofc_'+pc_t+'.xyzrgb')
                for pc_t in pc_types]
    pc_0_bound = (lig_pcds[0].get_max_bound()[0]-lig_pcds[0].get_min_bound()[0])*2
    # translate the images to put one between the other
    for i in range(1, len(lig_pcds)):
        lig_pcds[i] = lig_pcds[i].translate([pc_0_bound*i, 0, 0])
    # add the images to the list of views
    lig_views = lig_pcds
    # if vocab was informed
    # visualize the images colored by the labels below the previous ones, translate on the z axis
    if num_labels > 0:
        pc_0_bound = (lig_pcds[0].get_max_bound()[2] - lig_pcds[0].get_min_bound()[2]) * 3
        lig_labels = [read_pc_labels(pc_dir + '/' + entry + '/' + ligID + '_lig_pc_labels_'+pc_t+'.txt',
                                     class_mapping, num_labels).astype(str)
                      if pc_t not in ["qRankMask_5", "qRankMask_5_75_95", "qRankMask_5_7_9"] else
                      read_pc_labels(pc_dir + '/' + entry + '/' + ligID + '_lig_pc_labels_qRankMask.txt',
                                     class_mapping, num_labels).astype(str)
                        for pc_t in pc_types]
        lig_pcds_labels = [pcd.__copy__() for pcd in lig_pcds]
        for i in range(len(lig_pcds_labels)):
            if len(lig_labels[i]) > 0:
                np.asarray(lig_pcds_labels[i].colors)[:, :] = np.asarray([elements_color_set[x] for x in lig_labels[i]])
                lig_pcds_labels[i] = lig_pcds_labels[i].translate([0, 0, pc_0_bound])
        # add the images to the list of views
        lig_views = lig_views + lig_pcds_labels
    #
    # visualize the prediction if present
    if pred_directory is not None:
        pcd1 = o3d.io.read_point_cloud(pred_directory + '/' + ligID + '_predicted.xyzrgb')
        # np.asarray(pcd1.points)[:, :] = np.asarray(pcd1.points)
        # if the prediction exists, convert the classes index to a corresponding color
        if np.asarray(pcd1.points).shape[0] > 0:
            np.asarray(pcd1.colors)[:, :] = np.asarray([elements_color_set[str(int(x))] for x in np.asarray(pcd1.colors)[:, 0]])
            #[elements_color_set[x] for x in np.asarray(pcd1.colors)[:, 0]]
        # np.asarray(pcd1.points)[:, :] = np.asarray(pcd1.points) * grid_space
        pcd1 = pcd1.translate([0, 0, pc_0_bound * 2])
        # add the images to the list of views
        lig_views = lig_views + [pcd1]
    #
    # draw all the ligands view
    o3d.visualization.draw_geometries(lig_views)

def visualize_ligand_point_clound_feats_labels_prediction(lig_data_path, pc_dir, pred_directory, class_mapping_path,
                                                          vocab_path, pc_types):
    ligs_data = pd.read_csv(lig_data_path)
    # check if a vocabulary was informed, if not ignore the labels
    if vocab_path:
        # read mapping if present
        if class_mapping_path:
            mapping = pd.read_csv(class_mapping_path)
            mapping = mapping.sort_values('source')
            class_mapping = mapping.target.to_numpy()
            num_labels = len(mapping.target.unique())
            mapping = mapping[['mapping', 'target']][~mapping[['mapping', 'target']].duplicated()].sort_values('target')
            vocab = mapping.mapping.values
            # print(class_mapping)
            # print(vocab)
        else:
            class_mapping = None
            vocab = np.asarray([line.rstrip('\n') for line in open(vocab_path)] + ["solvent_background"])
            num_labels = len(vocab)
    else:
        num_labels = 0
        class_mapping = None
        
    # draw the pc img of each ligand, colored by the features, by the labels if vocab was informed and by the predictions if the predictions were informed
    for i in range(ligs_data.shape[0]):
        print("*** Ligand ID "+ ligs_data.loc[i,'ligID']+" ***")
        print("  - Ligand point clouds with features x target x predicted view")
        read_point_clouds_lig(ligs_data.loc[i,'ligID'], pc_dir, pred_directory, num_labels,
                              pc_types, class_mapping)


if __name__ == "__main__":
    # default parameters
    class_mapping_path = None
    pred_directory = None
    vocab_path = None
    if len(sys.argv) >= 4:
        lig_data_path = sys.argv[1]
        pc_dir = sys.argv[2]
        pc_types = sys.argv[3].split(',')
        if len(sys.argv) >= 5:
            vocab_path = sys.argv[4]        
            if len(sys.argv) >= 6:
                class_mapping_path = (sys.argv[5] if sys.argv[5] != 'none' else None)
                if len(sys.argv) >= 7:
                    pred_directory = (sys.argv[6] if sys.argv[6] != 'none' else None)
    else:
        sys.exit("Wrong number of parameters. There are three mandatory parameters to visualize a Lig-PCDB. \n"
                 "If the vocabulary is informed, the ligands' images colored by labels are draw. \n"
                 "If the predictions directory is informed, the ligands' images colored by the predicted labels are draw.\n"
                 "List of parameters:\n"
                 "1. list_ligands_path: A table with a list of ligands in CSV format containing the ligID column with "
                 "the ligands' ID that you want to visualize from the database;;\n"
                 "2. lig-pcdb_path: The path to the database folder where the ligands' images in point clouds are located;\n"
                 "3. img_types: The images types that you want to visualize, separated by comma and without spaces "
                 "(e.g. qRankMask_5,qRank0.95,qRank0.5);\n"
                 "4. vocab_path: (optional) The path to the vocabulary file used to label the database. "
                 "Default to 'none' - won't draw the ligands images colored by the labels;\n"                 
                 "5. class_mapping_path: (optional) The path to a class mapping file in CSV format or 'none'. "
                 "Default to 'none';\n"
                 "6. predictions_path: (optional) The path to a directory with the predictions result coming from the "
                 "np3_DL_segmentation module and organized in subfolders, one for each PDB entry that appear in the "
                 "ligands' list table, or 'none'. Default to 'none'.\n")

    visualize_ligand_point_clound_feats_labels_prediction(lig_data_path, pc_dir, pred_directory, class_mapping_path,
                                                          vocab_path, pc_types)

