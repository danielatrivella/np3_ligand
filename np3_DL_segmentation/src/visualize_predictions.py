import open3d as o3d
import pandas as pd
import sys, os
import numpy as np
from utils import elements_color_SP_test

def visualize_target_prediction(pred_directory, img_dir):
    entries_ious = pd.read_csv(pred_directory+'/entries_ious.csv')
    for i in range(entries_ious.shape[0]):
        print("*** Ligand ID "+ entries_ious.loc[i,'ligID']+" ***")
        print("  - Confusion matrix\n"+entries_ious.iloc[i,1:].to_string())
        print("  - Target view x Predicted view x Correct Mask - with and without background")
        # read the imgs from the lig-pcdb, when informed
        if img_dir is not None:
            entry = entries_ious.loc[i,'ligID'].split("_")[0]
            pcd_img_qrankMask = o3d.io.read_point_cloud(img_dir+'/'+entry+'/'+entries_ious.loc[i,'ligID']+'_lig_point_cloud_fofc_qRankMask_5.xyzrgb')
            np.asarray(pcd_img_qrankMask.points)[:, :] = np.asarray(pcd_img_qrankMask.points)
            # pcd_img_qrank5 = o3d.io.read_point_cloud(img_dir + '/' + entry + '/' + entries_ious.loc[
            #     i, 'ligID'] + '_lig_point_cloud_fofc_qRank0.5.xyzrgb')
            # np.asarray(pcd_img_qrank5.points)[:, :] = np.asarray(pcd_img_qrank5.points) / 0.5
            pcd_img_qrank75 = o3d.io.read_point_cloud(img_dir + '/' + entry + '/' + entries_ious.loc[
                i, 'ligID'] + '_lig_point_cloud_fofc_qRank0.75.xyzrgb')
            np.asarray(pcd_img_qrank75.points)[:, :] = np.asarray(pcd_img_qrank75.points)
            pcd_img_qrank95 = o3d.io.read_point_cloud(img_dir + '/' + entry + '/' + entries_ious.loc[
                i, 'ligID'] + '_lig_point_cloud_fofc_qRank0.95.xyzrgb')
            np.asarray(pcd_img_qrank95.points)[:, :] = np.asarray(pcd_img_qrank95.points)
        #
        # read the target and predicted imgs labels
        pcd = o3d.io.read_point_cloud(pred_directory+'/'+entries_ious.loc[i,'ligID']+'_target.xyzrgb')
        pcd1 = o3d.io.read_point_cloud(pred_directory + '/' + entries_ious.loc[i, 'ligID'] + '_predicted.xyzrgb')
        pcd2 = pcd1.__copy__()
        # compute the correctly predicted points and get the index of the points predicted as background
        correct_points = (np.asarray(pcd.colors) == np.asarray(pcd1.colors)).all(axis=1)
        np.asarray(pcd2.colors)[correct_points] = [0.75, 0.75, 0.75]
        np.asarray(pcd2.colors)[~correct_points] = [0.75, 0, 0]
        no_background_points = np.where(~(np.asarray(pcd1.colors) == [0,0,0]).all(axis=1))[0]
        # convert the target and predicted classes to colors
        np.asarray(pcd1.colors)[:, :] = np.asarray([elements_color_SP_test[str(int(x))] for x in np.asarray(pcd1.colors)[:, 0]])
        np.asarray(pcd.colors)[:, :] = np.asarray([elements_color_SP_test[str(int(x))] for x in np.asarray(pcd.colors)[:, 0]])
        # draw the point clouds
        if img_dir is not None:
            o3d.visualization.draw_geometries([pcd_img_qrankMask.translate([(pcd1.get_max_bound()-pcd1.get_min_bound())[0]*-4.5,
                                                               0, 0]),
                                               # pcd_img_qrank5.translate(
                                               #     [(pcd1.get_max_bound() - pcd1.get_min_bound())[0] * -4.5,
                                               #      0, 0]),
                                               pcd_img_qrank75.translate([(pcd1.get_max_bound()-pcd1.get_min_bound())[0]*-3,
                                                               0, 0]),
                                               pcd_img_qrank95.translate(
                                                   [(pcd1.get_max_bound() - pcd1.get_min_bound())[0] * -1.5,
                                                    0, 0]),
                                               pcd,
                                               pcd1.translate([(pcd1.get_max_bound()-pcd1.get_min_bound())[0]*1.5,
                                                               0, 0]),
                                               pcd2.translate([(pcd1.get_max_bound() - pcd1.get_min_bound())[0] * 3,
                                                               0, 0]),
                                               pcd1.select_by_index(no_background_points).translate(
                                                   # [0, 0,
                                                   #  (pcd1.get_max_bound() - pcd1.get_min_bound())[2] * 1.5]),  # desloca em z
                                                   [(pcd1.get_max_bound() - pcd1.get_min_bound())[0] * 3,
                                                    0, 0]),
                                               pcd2.select_by_index(no_background_points).translate(
                                                   # [0, 0, (pcd1.get_max_bound() - pcd1.get_min_bound())[2] * 1.5])])
                                                   [(pcd1.get_max_bound() - pcd1.get_min_bound())[0] * 3,
                                                    0, 0])])
        else:
            o3d.visualization.draw_geometries(
                [pcd,
                 pcd1.translate([(pcd1.get_max_bound() - pcd1.get_min_bound())[0] * 1.5,
                                 0, 0]),
                 pcd2.translate([(pcd1.get_max_bound() - pcd1.get_min_bound())[0] * 3,
                                 0, 0]),
                 pcd1.select_by_index(no_background_points).translate(
                     [0, 0,
                      (pcd1.get_max_bound() - pcd1.get_min_bound())[2] * 1.5]),
                 pcd2.select_by_index(no_background_points).translate(
                     [0, 0, (pcd1.get_max_bound() - pcd1.get_min_bound())[2] * 1.5])
                 ])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        img_dir = None
        pred_dir = sys.argv[1]
        if len(sys.argv) >= 3:
            img_dir = sys.argv[2]
    else:
        sys.exit("Wrong number of parameters. The predictions output directory path must be informed. \nOptionally the directory of the point cloud DB could also be informed as a second parameter.")

    visualize_target_prediction(pred_dir, img_dir)


