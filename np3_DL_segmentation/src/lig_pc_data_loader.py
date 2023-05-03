import torch, sys
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import logging
import numpy as np
from src.utils import random_rotate_3dpoints  #, draw_pc

SMOOTH = 1e-6
debug_img = False
# evalating img segm models https://www.jeremyjordan.me/evaluating-image-segmentation-models/
# https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
# different loss functions:https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

def compute_imbalance_ratio_H(class_representation, pc_type):
    class_representation = class_representation.astype('float')
    # compute imbalance ratio
    class_representation[class_representation > 0.0] = class_representation.max() / class_representation[class_representation > 0.0]
    # give a fixed weigh to the solvent class
    solvent_class_weight = (0.35 if pc_type.endswith("Mask") else 0.5)
    class_representation_tensor = torch.Tensor(np.concatenate([class_representation, [solvent_class_weight]])/solvent_class_weight)
    return class_representation_tensor

class LigandPointCloudDataset(Dataset):
    def __init__(self, ligs_retrieve_filepath, data_type, kfold, pc_path,
                 pc_type, vocab_path, class_mapping_path = None, rotation_rate = 0.5):
                 #,device_id=0, num_devices=1, num_points=0,
        # set the class attributes
        self.rotation_rate = rotation_rate
        self.pc_path = Path(pc_path)
        self.pc_type = pc_type
        self.data_type = data_type
        # self.num_points = num_points
        # read vocabulary labels and add background solvent class
        self.vocab = np.asarray([line.rstrip('\n') for line in open(vocab_path)] + ["solvent_background"])
        self.NUM_LABELS = len(self.vocab)
        # set vocab col names
        # vocab_cols = [str(x) for x in list(range(self.NUM_LABELS - 1))]
        # read the dataset
        #if self.pc_type in ['qRankMask', 'qRankMask_5', 'qRankMask_5_75_95', 'qRankMask_5_7_9']:
        #    pc_type_size_col = 'point_cloud_size_qRankMask'
        #else:
        #    pc_type_size_col = 'point_cloud_size_'+self.pc_type
        self.ligs_retrieve = pd.read_csv(ligs_retrieve_filepath,
                                         usecols = ['ligID', 'entry', 'grid_space', 'kfolds', 'test_val'])
                                                   #+[pc_type_size_col])  # + vocab_cols # not used anymore
        # check if the ligands files exists and if not remove the missing entries
        self.checkLigandsFiles()
        # not used anymore - compute initial class weights based on the class representation (total #atoms) imbalance ratio
        # self.class_H_imbalance_ratio = compute_imbalance_ratio_H(self.ligs_retrieve[vocab_cols].sum(0).values, pc_type)
        #
        # filter the data set entries depending on the data type and the kfold being used for testing and validation
        if data_type == 'train':
            self.ligs_retrieve = self.ligs_retrieve[self.ligs_retrieve.kfolds != kfold].reset_index(drop=True)
        else: # data type == 'val' or 'test'
            self.ligs_retrieve = self.ligs_retrieve[((self.ligs_retrieve.kfolds == kfold) &
                                           (self.ligs_retrieve.test_val == data_type))].reset_index(drop=True)
        #
        # read mapping if present
        if class_mapping_path:
            mapping = pd.read_csv(class_mapping_path)
            mapping = mapping.sort_values('source')
            self.class_mapping = mapping.target.to_numpy()
            self.NUM_LABELS = len(mapping.target.unique())
            mapping = mapping[['mapping', 'target']][~mapping[['mapping', 'target']].duplicated()].sort_values('target')
            self.vocab = mapping.mapping.values
            self.class_H_imbalance_ratio = np.asarray([1]*self.NUM_LABELS)
        else:
            # make mapping based on the classes occurrences if any == 0
            if any(self.class_H_imbalance_ratio == 0):
                # rm classes with no occurrence
                self.class_mapping = np.asarray(self.class_H_imbalance_ratio.tolist())
                self.NUM_LABELS = sum(self.class_mapping != 0)
                self.class_mapping[self.class_H_imbalance_ratio != 0] = list(range(self.NUM_LABELS))
                self.class_mapping[self.class_H_imbalance_ratio == 0] = -1
                self.vocab = self.vocab[self.class_H_imbalance_ratio != 0]
                self.class_H_imbalance_ratio = self.class_H_imbalance_ratio[self.class_H_imbalance_ratio != 0]
            else:
                self.class_mapping = None
        #
        # set the class attribute and drop unnecessary columns
        self.ligs_retrieve = self.ligs_retrieve[['ligID', 'entry', 'grid_space', 'kfolds', 'test_val']].sort_values(['kfolds', 'test_val'],
                                                                                                      ignore_index=True)
        # partition the dataset if num_devices > 1 - not used
        # self.partition(device_id, num_devices)
        #
        if self.ligs_retrieve.shape[0] == 0:
            sys.exit("No ligands present in this data set")
        else:
            logging.info('    => '+data_type+' ligands dataset created with '+str(self.ligs_retrieve.shape[0])+' entries.')
        # random shuffle the entries and random sort by the kfolds groups
        self.random_shuffle_kfolds()
    #
    def __len__(self):
        return self.ligs_retrieve.shape[0]
    #
    # def partition(self, device_id, num_devices): - not used
    #     if num_devices > 1:  # in a multi-process data loading
    #         # split workload
    #         per_worker = int(np.ceil((self.__len__() - 0) / num_devices))
    #         iter_start = 0 + device_id * per_worker
    #         iter_end = min(iter_start + per_worker, self.__len__())
    #         self.ligs_retrieve = self.ligs_retrieve.iloc[range(iter_start, iter_end),:].reset_index(drop=True)
        # self.indexes = iter(range(iter_start, iter_end))
    # check if the ligands input file exists an if not remove missing entries from the ligands table
    def checkLigandsFiles(self):
        logging.info('    ==> Checking if the ligands image and label files exists')
        self.ligs_retrieve['file_check'] = True
        n = self.__len__()
        for i in range(n):
            pcfile_xyzrgb = self.pc_path / self.ligs_retrieve.loc[i, 'entry'] / \
                            str(self.ligs_retrieve.loc[i, 'ligID'] + "_lig_point_cloud_fofc_" + self.pc_type + ".xyzrgb")
            if 'Mask' not in self.pc_type:
                pcfile_labels = self.pc_path / self.ligs_retrieve.loc[i, 'entry'] / \
                                str(self.ligs_retrieve.loc[i, 'ligID'] + "_lig_pc_labels_" + self.pc_type + ".txt")
            else:
                if self.pc_type in ['qRankMask', 'qRankMask_5', 'qRankMask_5_75_95', 'qRankMask_5_7_9']:
                    pcfile_labels = self.pc_path / self.ligs_retrieve.loc[i, 'entry'] / \
                                    str(self.ligs_retrieve.loc[i, 'ligID'] + "_lig_pc_labels_qRankMask.txt")
                else:
                    pcfile_labels = self.pc_path / self.ligs_retrieve.loc[i, 'entry'] / \
                                    str(self.ligs_retrieve.loc[i, 'ligID'] + "_lig_pc_labels_2sigmaMask.txt")
            if not pcfile_xyzrgb.exists() or not pcfile_labels.exists():
                self.ligs_retrieve.loc[i, 'file_check'] = False
        # remove files that are not present
        if ((~self.ligs_retrieve.file_check).any()):
            self.ligs_retrieve = self.ligs_retrieve[self.ligs_retrieve.file_check].reset_index(drop=True)
            self.ligs_retrieve.drop('file_check', axis='columns')
            n_rm = n - self.__len__()
            logging.info('      ==> Removed a total of ' + str(n_rm)+ ' ligands entries with missing input files')
        else:
            logging.info('      ==> OK')
        # check the entries size - not used
        # remove entries with less points than the num_points
        # if self.num_points > 0:
        #     logging.info(
        #         '    ==> Checking if the ligands point cloud size are greater or equal than the required number of points (' +
        #         str(self.num_points) + ')')
        #     if self.pc_type in ['qRankMask', 'qRankMask_5_75_95', 'qRankMask_5_7_9']:
        #         pc_type_size_col = 'point_cloud_size_qRankMask'
        #     else:
        #         pc_type_size_col = 'point_cloud_size_' + self.pc_type
        #     if any(self.ligs_retrieve[pc_type_size_col] < self.num_points):
        #         logging.info('      ==> Removed a total of ' + str(
        #                 sum(self.ligs_retrieve[pc_type_size_col] < self.num_points)) +
        #                            ' ligands entries with a point cloud size < ' + str(self.num_points))
        #         self.ligs_retrieve = self.ligs_retrieve[self.ligs_retrieve[pc_type_size_col] >= self.num_points]
        #     else:
        #         logging.info('      ==> OK')
    def __getitem__(self, i):
        # print("@@@ Get item ",i, " ligCode", self.ligs_retrieve.loc[i,'ligID'])
        pcfile_xyzrgb = self.pc_path / self.ligs_retrieve.loc[i,'entry'] / \
                        str(self.ligs_retrieve.loc[i,'ligID']+"_lig_point_cloud_fofc_"+self.pc_type+".xyzrgb")
        if 'Mask' not in self.pc_type:
            pcfile_labels = self.pc_path / self.ligs_retrieve.loc[i, 'entry'] / \
                            str(self.ligs_retrieve.loc[i, 'ligID'] + "_lig_pc_labels_" + self.pc_type + ".txt")
        else:
            if self.pc_type in ['qRankMask', 'qRankMask_5', 'qRankMask_5_75_95', 'qRankMask_5_7_9']:
                pcfile_labels = self.pc_path / self.ligs_retrieve.loc[i, 'entry'] / \
                                str(self.ligs_retrieve.loc[i, 'ligID'] + "_lig_pc_labels_qRankMask.txt")
            else:
                pcfile_labels = self.pc_path / self.ligs_retrieve.loc[i, 'entry'] / \
                                str(self.ligs_retrieve.loc[i, 'ligID'] + "_lig_pc_labels_2sigmaMask.txt")
        # read coords and feats
        coords, feats = self.read_pc_table(pcfile_xyzrgb)
        # Preprocess input
        # normalize color - Convert color in range [0, 1] to [-0.5, 0.5].
        feats -= 0.5
        labels = self.read_pc_labels(pcfile_labels)
        if coords.shape[0] != labels.shape[0]:
            sys.exit("Error matching length of coords and labels from dataset entry "+str(i)+
                     " - ligID "+self.ligs_retrieve.loc[i,'ligID'])
        #
        # randomly rotate input coords
        if self.data_type == 'train' and np.random.rand() < self.rotation_rate:
            coords = random_rotate_3dpoints(coords)
        # draw pc to visualize data
        # if debug_img:
        #     draw_pc(coords, feats)
        # removed num_points - tested for the DGCNN, but not used
        # if self.num_points > 0:
        #     # random sub sample the input to return a point cloud with a number of points equals the provided num_points
        #     # sample num_points for this entry
        #     mapping = np.random.choice(coords.shape[0], size=self.num_points, replace=False)
        #     return torch.FloatTensor(np.concatenate([coords[mapping],feats[mapping]], 1)), torch.LongTensor(labels[mapping])
        # else:
        # Quantize the input
        quantized_coords, mapping = ME.utils.sparse_quantize(coordinates=np.floor(coords / self.ligs_retrieve.loc[i,'grid_space']),
                                                             return_index=True)
        # draw pc to visualize data after quantize
        # if debug_img:
        #     draw_pc(quantized_coords[mapping], feats[mapping])
        # return {
        #     "coordinates": quantized_coords,
        #     "features": feats[mapping],
        #     "labels": labels[mapping],
        # }
        return quantized_coords, feats[mapping], labels[mapping]
    def get_entry_id(self, i):
        return self.ligs_retrieve.loc[i,'ligID']
    def get_grid_space(self):
        return self.ligs_retrieve.grid_space.unique()[0]
    def read_pc_table(self, pcfile):
        pc = pd.read_table(pcfile, header=None, delimiter=" ")
        # if self.pc_type in ['sigmaMask', 'qRankMask_5_75_95'] return the three channels as features
        # else only return one of the channels as features
        return np.ascontiguousarray(pc.iloc[:, :3].values, np.float64), pc.iloc[:, 3:(3+self.num_feats())].values.astype(np.float64)
    # the labels file is expected to have one label per row
    def read_pc_labels(self, pcfile_labels):
        labels = pd.read_table(pcfile_labels, header=None, delimiter=" ")[0].values.astype(np.int32)
        solvent_mask = (labels == -1)
        # map labels using the given mapping table where is not solvent label
        if self.class_mapping is not None:
            labels[~solvent_mask] = self.class_mapping[labels[~solvent_mask]]
            labels[solvent_mask] = self.class_mapping[-1] # last target is always background
        else:
            labels[solvent_mask] = self.NUM_LABELS - 1  # solvent_background class
        #
        return labels
    def num_feats(self):
        if self.pc_type in ['sigmaMask', 'qRankMask_5_75_95', 'qRankMask_5_7_9']:
            # return the three channels as features - these images are not used anymore
            return 3
        else:  # only return one of the channels as features
            return 1
    def num_classes(self):
        return self.NUM_LABELS
    def get_classnames(self):
        return self.vocab
    # def get_lr_decrease_index(self):
    #     return self.lr_decrease_i
    def get_class_representation_ratio(self):
        return self.class_H_imbalance_ratio # to be used and the weights initialization
    def random_shuffle_kfolds(self):
        logging.info('    ==> Random shuffle the dataset and random sort by the kfolds group')
        entries_order = self.ligs_retrieve.index.values
        np.random.shuffle(entries_order)
        ligs_retrieve = self.ligs_retrieve.iloc[entries_order, :]
        ligs_kfolds_groups = [df for _, df in ligs_retrieve.groupby('kfolds')]
        np.random.shuffle(ligs_kfolds_groups)
        self.ligs_retrieve = pd.concat(ligs_kfolds_groups).reset_index(drop=True)


def collation_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))
    # coords_batch, feats_batch, labels_batch = [], [], []
    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)
    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).long()
    return coords_batch, feats_batch, labels_batch
# def collation_fn(list_data):
#     r"""
#         Collation function for MinkowskiEngine.SparseTensor that creates batched
#         cooordinates given a list of dictionaries.
#         """
#     coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
#         [d["coordinates"] for d in list_data],
#         [d["features"] for d in list_data],
#         [d["labels"] for d in list_data],
#         dtype=torch.float32,
#     )
#     return {
#         "coordinates": coordinates_batch,
#         "features": features_batch,
#         "labels": labels_batch,
#     }

def ligands_dataloader(config, data_type):  #, device_id=0, num_devices=1):
    np.random.seed(config.seed)
    # Dataset, data loader
    if data_type == 'train':
        lig_dataset = LigandPointCloudDataset(config.ligs_data_filepath, data_type, config.kfold, config.lig_pcdb_path,
                                              config.pc_type, config.vocab_path,
                                              config.class_mapping_path,
                                              #device_id=device_id,num_devices=num_devices,
                                              # num_points=config.num_points,
                                              rotation_rate=config.rotation_rate)
        # if config.num_points > 0:
        #     # DGCNN - not used
        #     lig_dataloader = DataLoader(
        #         lig_dataset,
        #         batch_size=config.batch_size,
        #         num_workers=config.num_workers,
        #         shuffle=True)
        # else:
        # ME
        lig_dataloader = DataLoader(
            lig_dataset,
            batch_size=config.batch_size,
            collate_fn=collation_fn,
            # 2) collate_fn=ME.utils.batch_sparse_collate,
            # collate_fn=ME.utils.SparseCollation(),
            # collate_fn=ME.utils.batch_sparse_collate,
            num_workers=config.num_workers,
            shuffle=True)
    else:
        lig_dataset = LigandPointCloudDataset(config.ligs_data_filepath, data_type, config.kfold, config.lig_pcdb_path,
                                              config.pc_type, config.vocab_path,
                                              config.class_mapping_path,
                                              #False,device_id=device_id,num_devices=num_devices,
                                              # num_points=config.num_points,
                                              rotation_rate=config.rotation_rate)
        # if config.num_points > 0:
        #     # DGCNN - not used
        #     lig_dataloader = DataLoader(
        #         lig_dataset,
        #         batch_size=config.batch_size,
        #         num_workers=config.num_val_workers)
        # else:
        # ME
        lig_dataloader = DataLoader(
            lig_dataset,
            batch_size=(config.val_batch_size if data_type == 'val' else config.test_batch_size),
            collate_fn=collation_fn,
            num_workers=config.num_val_workers)
    #
    return lig_dataloader

if __name__ == '__main__':
    pass
