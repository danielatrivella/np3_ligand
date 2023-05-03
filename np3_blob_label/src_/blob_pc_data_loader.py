import torch, sys
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import logging
import numpy as np


class BlobPointCloudDataset(Dataset):
    def __init__(self, blob_retrieve_filepath, pc_path, pc_type):
        # set the class attributes
        self.pc_path = Path(pc_path)  # blobs_img folder
        self.pc_type = pc_type # which image to use
        self.blobs_retrieve = pd.read_csv(blob_retrieve_filepath,
                                         usecols = ['blobID', 'grid_space', 'x', 'y', 'z'])
        # check if the blobs files exists and if not remove the missing entries
        self.checkBlobsFiles()
        #
        if self.blobs_retrieve.shape[0] == 0:
            logging.info("No blobs present in this data set")
            return None
        else:
            logging.info('    => '+pc_type+' blobs dataset created with '+
                         str(self.blobs_retrieve.shape[0])+' entries.')
    #
    def __len__(self):
        return self.blobs_retrieve.shape[0]
    # check if the blobs input file exists an if not remove missing entries from the blobs table
    def checkBlobsFiles(self):
        logging.info('    ==> Checking if the blobs image files exists')
        self.blobs_retrieve['file_check'] = True
        n = self.__len__()
        for i in range(n):
            pcfile_xyzrgb = self.pc_path / self.blobs_retrieve.loc[i, 'blobID'] / \
                            str(self.blobs_retrieve.loc[i, 'blobID'] + "_point_cloud_fofc_" + self.pc_type + ".xyzrgb")
            if not pcfile_xyzrgb.exists():
                self.blobs_retrieve.loc[i, 'file_check'] = False
        # remove files that are not present from the blobs list
        if ((~self.blobs_retrieve.file_check).any()):
            self.blobs_retrieve = self.blobs_retrieve[self.blobs_retrieve.file_check].reset_index(drop=True)
            self.blobs_retrieve.drop('file_check', axis='columns')
            n_rm = n - self.__len__()
            logging.info('      ==> Removed a total of ' + str(n_rm)+ ' blobs entries with missing input files')
        else:
            logging.info('      ==> OK')
    def __getitem__(self, i):
        # print("@@@ Get item ",i, " blobCode", self.blobs_retrieve.loc[i,'blobID'])
        pcfile_xyzrgb = self.pc_path / self.blobs_retrieve.loc[i,'blobID'] / \
                        str(self.blobs_retrieve.loc[i,'blobID']+"_point_cloud_fofc_"+self.pc_type+".xyzrgb")
        # read coords and feats
        coords, feats = self.read_pc_table(pcfile_xyzrgb)
        # Preprocess input
        # normalize color - Convert color in range [0, 1] to [-0.5, 0.5].
        feats -= 0.5
        # Quantize the input
        quantized_coords, mapping = ME.utils.sparse_quantize(coordinates=np.floor(coords / self.blobs_retrieve.loc[i,'grid_space']),
                                                             return_index=True)
        return quantized_coords, feats[mapping],
    def get_entry_id(self, i):
        return self.blobs_retrieve.loc[i, 'blobID']
    def read_pc_table(self, pcfile):
        pc = pd.read_table(pcfile, header=None, delimiter=" ")
        # if self.pc_type in ['sigmaMask', 'qRankMask_5_75_95'] return the three channels as features
        # else only return one of the channels as features
        return np.ascontiguousarray(pc.iloc[:, :3].values, np.float64), pc.iloc[:, 3:(3+self.num_feats())].values.astype(np.float64)
    def num_feats(self):
        if self.pc_type in ['sigmaMask', 'qRankMask_5_75_95', 'qRankMask_5_7_9']:  # return the three channels as features
            return 3
        else:  # only return one of the channels as features
            return 1


def collation_fn(data_labels):
    coords, feats = list(zip(*data_labels))
    # coords_batch, feats_batch, labels_batch = [], [], []
    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)
    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    return coords_batch, feats_batch


def blobs_dataloader(blobs_data_filepath, pc_path, data_type, batch_size, num_workers):
    np.random.seed(123)
    # Dataset, data loader
    blob_dataset = BlobPointCloudDataset(blobs_data_filepath, pc_path, data_type)
    if not blob_dataset:
        return None
    # ME
    blob_dataloader = DataLoader(
        blob_dataset,
        batch_size=batch_size,
        collate_fn=collation_fn,
        num_workers=num_workers)
    #
    return blob_dataloader
