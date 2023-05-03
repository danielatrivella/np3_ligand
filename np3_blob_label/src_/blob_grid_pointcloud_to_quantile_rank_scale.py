from pathlib import Path
import numpy as np
import open3d as o3d
import sys
import pandas as pd
import time
from p_tqdm import p_map
from multiprocessing import cpu_count
import logging

test_mode = False

if test_mode:
    # import seaborn as sns
    import matplotlib.pyplot as plt

# def plot_dist(data):
#     sns.set_theme(style="darkgrid")
#     sns.displot(data, height=5)
#     # plt.show()
#     plt.savefig('distribution_rho.png')

# the Van der Waals Radii of Elements from  S. S. Batsanov 2001
# elements_radii = {'B': 1.8, 'C': 1.7,
#                   'N': 1.6, 'P': 1.95,
#                   'O': 1.55, 'S': 1.8, 'Se': 1.9,
#                   'Cl': 1.8, 'F': 1.5, 'Br': 1.9, 'I': 2.1}
# the experimental Van der Waals Radii averaged from 1.5A to 1.8A rounded in 2 decimals from XGen 2020
# - B received S and Se received Br radii
# elements_radii = {'B': 1.4, 'C': 1.46,
#                   'N': 1.44, 'P': 1.4,
#                   'O': 1.42, 'S': 1.4, 'Se': 1.37,
#                   'Cl': 1.4, 'F': 1.4, 'Br': 1.37, 'I': 1.37}
raddi_weight = 0.65
elements_radii_w_reso = {'1.5': {'B': 1.3363*raddi_weight, 'C': 1.3960*raddi_weight,
                                 'N': 1.3774*raddi_weight, 'P': 1.3315*raddi_weight,
                                 'O': 1.3576*raddi_weight, 'S': 1.3363*raddi_weight, 'Se': 1.3036*raddi_weight,
                                 'Cl': 1.3381*raddi_weight, 'F': 1.3405*raddi_weight, 'Br': 1.3036*raddi_weight,
                                 'I': 1.3003*raddi_weight},
                         '1.6': {'B': 1.3777 * raddi_weight, 'C': 1.4329 * raddi_weight,
                                 'N': 1.4140 * raddi_weight, 'P': 1.3735 * raddi_weight,
                                 'O': 1.3945 * raddi_weight, 'S': 1.3777 * raddi_weight, 'Se': 1.3459 * raddi_weight,
                                 'Cl': 1.3786 * raddi_weight, 'F': 1.3783 * raddi_weight, 'Br': 1.3459 * raddi_weight,
                                 'I': 1.3429 * raddi_weight},
                         '1.7': {'B': 1.4212*raddi_weight, 'C': 1.4728*raddi_weight,
                                 'N': 1.4536*raddi_weight, 'P': 1.4179*raddi_weight,
                                 'O': 1.4347*raddi_weight, 'S': 1.4212*raddi_weight, 'Se': 1.3906*raddi_weight,
                                 'Cl': 1.4218*raddi_weight, 'F': 1.4191*raddi_weight, 'Br': 1.3906*raddi_weight,
                                 'I': 1.3876*raddi_weight},
                         '1.8': {'B': 1.4713*raddi_weight, 'C': 1.5184*raddi_weight,
                                 'N': 1.4992*raddi_weight, 'P': 1.4686*raddi_weight,
                                 'O': 1.4812*raddi_weight, 'S': 1.4713*raddi_weight, 'Se': 1.4413*raddi_weight,
                                 'Cl': 1.4710*raddi_weight, 'F': 1.4665*raddi_weight, 'Br': 1.4413*raddi_weight,
                                 'I': 1.4383*raddi_weight},
                         '1.9': {'B': 1.5193*raddi_weight, 'C': 1.5634*raddi_weight,
                                 'N': 1.5442*raddi_weight, 'P': 1.5175*raddi_weight,
                                 'O': 1.5268*raddi_weight, 'S': 1.5193*raddi_weight, 'Se': 1.4902*raddi_weight,
                                 'Cl': 1.5184*raddi_weight, 'F': 1.5127*raddi_weight, 'Br': 1.4902*raddi_weight,
                                 'I': 1.4878*raddi_weight},
                         '2.0': {'B': 1.5748*raddi_weight, 'C': 1.6159*raddi_weight,
                                 'N': 1.5967*raddi_weight, 'P': 1.5739*raddi_weight,
                                 'O': 1.5799*raddi_weight, 'S': 1.5748*raddi_weight, 'Se': 1.5472*raddi_weight,
                                 'Cl': 1.5739*raddi_weight, 'F': 1.5664*raddi_weight, 'Br': 1.5472*raddi_weight,
                                 'I': 1.5448*raddi_weight},
                         '2.1': {'B': 1.6252*raddi_weight, 'C': 1.6636*raddi_weight,
                                 'N': 1.6444*raddi_weight, 'P': 1.6246*raddi_weight,
                                 'O': 1.6282*raddi_weight, 'S': 1.6252*raddi_weight, 'Se': 1.5979*raddi_weight,
                                 'Cl': 1.6234*raddi_weight, 'F': 1.6153*raddi_weight, 'Br': 1.5979*raddi_weight,
                                 'I': 1.5955*raddi_weight},
                         '2.2': {'B': 1.6813*raddi_weight, 'C': 1.7173*raddi_weight,
                                 'N': 1.6984*raddi_weight, 'P': 1.6810*raddi_weight,
                                 'O': 1.6825*raddi_weight, 'S': 1.6813*raddi_weight, 'Se': 1.6549*raddi_weight,
                                 'Cl': 1.6792*raddi_weight, 'F': 1.6702*raddi_weight, 'Br': 1.6549*raddi_weight,
                                 'I': 1.6525*raddi_weight}
                         }

expansion_radii = np.round(max(list(elements_radii_w_reso['2.2'].values())), 1)


# filename is a Path var, return number of lines
def rawcount(filename):
    if not filename.exists():
        return 0
    f = open(filename.as_posix(), 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read
    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)
    return lines


def min_max(x):
    return np.min(x), np.max(x)


def min_max_scale(x, min_scale, max_scale):
    return (x - min_scale) / (max_scale - min_scale)


blob_center_cube_dist = 1.0  # equals to the search cube dimension in the find blobs script
# return the points indexes of the point cloud blob_grid_fofc connected to all x,y,z position in the blob center cube
# which is the center position plus the center position with translations of 1.5A in 6 horizontal and vertical positions
# and 8 transversal position (correspond to the center cube sides and corners
# where search_radius = grid_space * 1.42 + 0.15
def extract_blob_point_cloud(blob_grid_fofc, blob_center, grid_space):
    # print("Extract the blob point cloud cluster using its center cube position")
    blob_grid_fofc_kdtree = o3d.geometry.KDTreeFlann(blob_grid_fofc)
    # compute the center cube position, transfer the center to a cube with 1.5A distance to the sides and corners
    blob_center = np.asarray(blob_center)
    blob_center_cube_pos = [blob_center + adj for adj in
                       [[0, 0, 0], [blob_center_cube_dist, 0, 0], [0, blob_center_cube_dist, 0],
                        [0, 0, blob_center_cube_dist], [-blob_center_cube_dist, 0, 0], [0, -blob_center_cube_dist, 0],
                        [0, 0, -blob_center_cube_dist]] +
                       [[x, y, z] for x in [blob_center_cube_dist, -blob_center_cube_dist] for y in
                        [blob_center_cube_dist, -blob_center_cube_dist] for z in [blob_center_cube_dist, -blob_center_cube_dist]]]
    # apply dijkstra algorithm using the kdtree search from each blob center cube position if necessary
    # (if the ligand density is connected one position will be enough).
    # It uses a visited binary array to store the reached points and a reached list to store the next points to be expanded
    search_radius = grid_space * 1.42 + 0.15  # gets the corners of the point multiplying by ~(2)^1/2 +
                                              # a gap to deal with missing points in valley regions of the electron density
    points_visited = np.zeros(len(blob_grid_fofc.points))  # 1 for reached, 2 for expanded, 0 for not seen yet
    # search the blob density around the center cube positions and expand pc
    for center_cube_pos in blob_center_cube_pos:
        points_reached = np.asarray(blob_grid_fofc_kdtree.search_radius_vector_3d(center_cube_pos, radius=search_radius)[1])
        # if the point was already expand continue, else remove the expanded ones and proceed for the missing one
        if (points_visited[points_reached] == 2).all():
            continue
        else:
            points_reached = points_reached[points_visited[points_reached] != 2]
        # set the initial not seen points in the neighborhood of the current position
        points_visited[points_reached] = 1
        points_reached = list(points_reached)
        # expand each reachable point from the current center_cube position
        while len(points_reached) > 0:
            expand_point = points_reached.pop()
            points_visited[expand_point] = 2
            neighbors = np.asarray(
                blob_grid_fofc_kdtree.search_radius_vector_3d(blob_grid_fofc.points[expand_point],
                                                             radius=search_radius)[1])
            neighbors = neighbors[points_visited[neighbors] == 0]
            points_visited[neighbors] = 1
            points_reached.extend(neighbors)
    # extract the expanded points indexes
    blob_points_idx = np.where(points_visited == 2)[0]
    return blob_points_idx



# extract the blob point cloud for a given quantile rank contour value and
# return the points indexes in the given blob_pc_pruned
def create_blob_pc_q_rank(blob_pc_mask, blob_center, q_rank_contour, grid_space, blob_output_path, blobID):
    if q_rank_contour is not None:
        file_label = str(q_rank_contour)
        if q_rank_contour not in ["Mask_5_75_95", "Mask_5_7_9", "Mask_5"]:
            # apply a contour
            blob_pc_qrank, pc_qrank_idx = extract_pc_gt_value(blob_pc_mask, q_rank_contour)
            # if selected less points, extract blob pc again to remove isolated noise
            if len(pc_qrank_idx) < len(blob_pc_mask.points):
                # extract blob point cloud idx
                blob_sel = extract_blob_point_cloud(blob_pc_qrank, blob_center, grid_space)
                blob_pc_mask = blob_pc_qrank.select_by_index(blob_sel)
                pc_qrank_idx = pc_qrank_idx[blob_sel]
    else:
        file_label = "Mask"
        q_rank_contour = np.min(np.asarray(blob_pc_mask.colors)[:, 0])
    # min_max scale image if it is not the image with different scales in each chanel
    if file_label not in ["Mask_5_75_95", "Mask_5_7_9", "Mask_5"]:
        np.asarray(blob_pc_mask.colors)[:, :] = min_max_scale(np.asarray(blob_pc_mask.colors)[:, :],
                                                              q_rank_contour,
                                                              1.0)
    if test_mode:
        o3d.visualization.draw_geometries([blob_pc_mask])
        # plot_dist(np.asarray(blob_pc_mask.colors)[:, 0])
        # if file_label in ["Mask_5_75_95", "Mask_5_7_9"]:
        #     plot_dist(np.asarray(blob_pc_mask.colors)[:, 1])
        #     plot_dist(np.asarray(blob_pc_mask.colors)[:, 2])
    # save the blob point cloud
    o3d.io.write_point_cloud(
        (blob_output_path / (blobID + "_point_cloud_fofc_qRank"+file_label+".xyzrgb")).as_posix(),
        blob_pc_mask)
    #
    # return the indexes of the contour and their scaled intensity values
    if file_label not in ["Mask", "Mask_5_75_95", "Mask_5_7_9", "Mask_5"]:
        return pc_qrank_idx, np.asarray(blob_pc_mask.colors)[:, 0]
    else:
        return None

def extract_pc_gt_value(pc, value, idx=True):
    # select values above the contour
    pc_sel_idx = np.where(np.asarray(pc.colors)[:, 0] > value)[0]
    pc_sel = pc.select_by_index(pc_sel_idx)
    if idx:
        return pc_sel, pc_sel_idx
    else:
        return pc_sel


# extract the blob point cloud for a given quantile rank contour value and return it
def extract_blob_pc_Q_rank_idx(blob_grid, q_rank_contour, blob_center, grid_space):
    # print("Extract the blob point cloud for",q_rank_contour, "quantile rank")
    # select values above the contour
    blob_q_rank, blob_q_rank_sel_idx = extract_pc_gt_value(blob_grid, q_rank_contour)
    if len(blob_q_rank.points) > 0:
        # extract blob point cloud
        blob_sel = extract_blob_point_cloud(blob_q_rank, blob_center, grid_space)
        if test_mode:
            o3d.visualization.draw_geometries([blob_q_rank.select_by_index(blob_sel)])
    else:
        return None
    # return the extracted blob pc indexes in the given contour
    return blob_q_rank_sel_idx[blob_sel]


def get_boundary_idx(pc, grid_space=0.2):
    # print("Get the point cloud boundary indexes")
    pc_kdtree = o3d.geometry.KDTreeFlann(pc)
    # store the point indexes that have less than 5 neighbours
    boundary_idx = []
    for i, point in enumerate(np.asarray(pc.points)):
        if len(pc_kdtree.search_radius_vector_3d(point, 1.1*grid_space)[1]) < 6:
            boundary_idx.append(i)
    # return their indexes
    return boundary_idx


def extract_pc_neighborhood(pc, mask_idx, search_radius):
    # print("Extract the point cloud neighborhood from the grid by expanding", search_radius,
    #       "A from the boundary points of the provided pc mask and indexes")
    pc_mask = pc.select_by_index(mask_idx)
    # get the boundary points from the pc maks
    boundary_points = np.asarray(pc_mask.points)[get_boundary_idx(pc_mask)]
    # expand the boundary points in <seach_radius> A
    pc_kdtree = o3d.geometry.KDTreeFlann(pc)
    #
    # Expands each point neighborhood using the kdtree search
    # It uses a visited binary array to store the reached points
    points_visited = np.zeros(len(pc.points))  # 1 for reached 0 for not seen yet
    for point in boundary_points:
        points_reached = np.asarray(pc_kdtree.search_radius_vector_3d(point, radius=search_radius)[1])
        # if the point was already visited continue, else remove the visited ones and proceed for the missing one
        if (points_visited[points_reached] == 1).all():
            continue
        else:
            points_reached = points_reached[points_visited[points_reached] != 1]
        # set the not seen points in the neighborhood of the points
        points_visited[points_reached] = 1
    # extract the expanded points indexes
    ligand_points_idx = np.where(points_visited == 1)[0]
    # merge this list of indexes with the list of indexes of the mask
    ligand_points_idx = np.unique(np.concatenate((ligand_points_idx, mask_idx), 0))
    return pc.select_by_index(ligand_points_idx)


# histogram equalization
def quantile_rank_scale_fast(p_grid_values):
    # for each point in the grid count N(pj)=number of points with p(n) < pj
    # equivalent to its position in the sorted array, ties placed on the left
    N_p_grid = np.searchsorted(p_grid_values, p_grid_values, side='left', sorter=np.argsort(p_grid_values))
    N_grid = p_grid_values.size
    # compute the quantile rank q = N(pj)/Ngrid, where N(pj)=number of points with p(n) < pj
    q_grid_values = N_p_grid / N_grid
    # return density values rank scaled in the ligand grid
    return q_grid_values


def create_blob_point_cloud_quantile_rank_scaled(blob_data_row, blobs_grid_path, output_path, overwrite_pc, verbose=False):
    # i = blob_data_row[0]
    blob_data = blob_data_row[1]
    blob_out = pd.Series({'blobID': blob_data['blobID']})
    blob_out["point_cloud"] = False
    blob_out["point_cloud_size"] = -1  # 0.95 quantile rank mask expanded
    blob_out["point_cloud_size_qRank0.5"] = -1
    blob_out["point_cloud_size_qRank0.7"] = -1
    blob_out["point_cloud_size_qRank0.75"] = -1
    blob_out["point_cloud_size_qRank0.8"] = -1
    blob_out["point_cloud_size_qRank0.85"] = -1
    blob_out["point_cloud_size_qRank0.9"] = -1
    blob_out["point_cloud_size_qRank0.95"] = -1
    blob_out["missing_blob"] = "False"
    blob_out['time_process'] = None
    #
    # create the grid and point cloud for each valid blob
    #
    # validate blob output path
    blob_id = blob_data['blobID']
    grid_space = blob_data["grid_space"]
    blob_center = [blob_data["x"], blob_data["y"], blob_data["z"]]
    blob_grid_file = blobs_grid_path / str(blob_id+'_grid_point_cloud_fofc.xyzrgb')
    blob_output_path = output_path / blob_id

    if not blob_grid_file.exists():
        # print("Warning: The Blob entry ", blob_id,
        #       "grid pointclouds was not created.")
        blob_out["missing_blob"] = "ErrorBlobPcGridNoData"
        return blob_out

    # for each blob in the current entry, scale the electron density values of their grid to the quantile rank,
    # extract a mask with 0.95 contour of the blob pc with expansion and
    # create the point clouds in others quantiles contours inside the mask

    # if the blob was already processed and overwrite is false, only get the pcs sizes and skip the blob processing
    if (blob_output_path / (blob_id + "_point_cloud_fofc_qRankMask.xyzrgb")).exists() and not overwrite_pc:
        # print("* Already processed blob", blob_id, i + 1, "/", n, ". Skipping to next entry. *")
        blob_out["point_cloud"] = True
        blob_out["point_cloud_size"] = rawcount(
            (blob_output_path / (blob_id + "_point_cloud_fofc_qRankMask.xyzrgb")))
        blob_out["point_cloud_size_qRank0.5"] = rawcount(
            (blob_output_path / (blob_id + "_point_cloud_fofc_qRank0.5.xyzrgb")))
        blob_out["point_cloud_size_qRank0.7"] = rawcount(
            (blob_output_path / (blob_id + "_point_cloud_fofc_qRank0.7.xyzrgb")))
        blob_out["point_cloud_size_qRank0.75"] = rawcount(
            (blob_output_path / (blob_id + "_point_cloud_fofc_qRank0.75.xyzrgb")))
        blob_out["point_cloud_size_qRank0.8"] = rawcount(
            (blob_output_path / (blob_id + "_point_cloud_fofc_qRank0.8.xyzrgb")))
        blob_out["point_cloud_size_qRank0.85"] = rawcount(
            (blob_output_path / (blob_id + "_point_cloud_fofc_qRank0.85.xyzrgb")))
        blob_out["point_cloud_size_qRank0.9"] = rawcount(
            (blob_output_path / (blob_id + "_point_cloud_fofc_qRank0.9.xyzrgb")))
        blob_out["point_cloud_size_qRank0.95"] = rawcount(
            (blob_output_path / (blob_id + "_point_cloud_fofc_qRank0.95.xyzrgb")))
        if blob_out["point_cloud_size_qRank0.75"] == 0:
            # print("Warning: No points for blob", blob_id, "in 0.75 quantile rank contour")
            blob_out["missing_blob"] = "ErrorPc075qRankParsing"
        if blob_out["point_cloud_size_qRank0.95"] == 0:
            # print("Warning: No points for blob", blob_id, "in 0.95 quantile rank contour")
            blob_out["missing_blob"] = "ErrorPc095qRankParsing"
        return blob_out

    # create the blob output path if it does not exists
    blob_output_path.mkdir(parents=True, exist_ok=True)
    # round the pdb entry resolution
    # reso = str(round(float(blob_data['resolution']),1))
    # print("\n* Start processing blob entry", blob_id, "-", i + 1, "/", n, "*")
    t1 = time.time()
    # start processing the blobs grids, use the center value to search for the blobs residual density map
    try:
        # read grid point cloud fofc and scale the density values using the quantile rank
        blob_grid = o3d.io.read_point_cloud(blob_grid_file.as_posix())
        rho_Q_rank = quantile_rank_scale_fast(p_grid_values=np.asarray(blob_grid.colors)[:, 0])
        # print("Number of points in the blob grid", len(blob_grid.points))
        np.asarray(blob_grid.colors)[:, 0] = rho_Q_rank
        np.asarray(blob_grid.colors)[:, 1] = rho_Q_rank
        np.asarray(blob_grid.colors)[:, 2] = rho_Q_rank
        del rho_Q_rank

        # extract the blob point cloud using a mask with 0.95 quantile rank expanded to remove background noise
        # before applying other contour levels
        # expand from the blob center
        blob_fofc_idx = extract_blob_pc_Q_rank_idx(blob_grid, 0.95, blob_center, grid_space)
        # check if the number of points at 0.95 contour is below 20% of the number of points of the smaller atom sphere
        # if yes skip this entry
        if len(blob_fofc_idx) < (4 / 3 * np.pi * np.power(min(elements_radii_w_reso['1.5'].values()), 3) / np.power(grid_space, 3)):
            raise Exception
        blob_fofc = extract_pc_neighborhood(blob_grid, blob_fofc_idx, expansion_radii)
        #
        # print("Number of points in the blob point cloud 0.95 quantile rank mask", len(blob_fofc.points))
        # draw the geometries
        if test_mode:
            # print("Plot rho grid points distribution and the blob pc in 0.95 quantile rank mask expanded")
            # plot_dist(np.asarray(blob_grid.colors)[:, 0])
            o3d.visualization.draw_geometries([blob_fofc])
        del blob_grid, blob_fofc_idx
    except Exception as e:
        # print("Error message:", e)
        blob_out["missing_blob"] = "ErrorPc095qRankMaskCreation"
        return blob_out

    # set the mask image size
    blob_out["point_cloud_size"] = len(blob_fofc.points)
    ## create the blob images with different contours
    # create the blob point cloud in quantile rank scaled with the 0.95 quantile mask expanded, and
    # with 0.5, 0.75 and 0.95 contour inside the mask
    try:
        create_blob_pc_q_rank(blob_fofc, blob_center, None, grid_space, blob_output_path, blob_id)
        idx_5, rho_5 = create_blob_pc_q_rank(blob_fofc, blob_center, 0.5, grid_space, blob_output_path, blob_id)
        idx_7, rho_7 = create_blob_pc_q_rank(blob_fofc, blob_center, 0.7, grid_space, blob_output_path, blob_id)
        idx_75, rho_75 = create_blob_pc_q_rank(blob_fofc, blob_center, 0.75, grid_space, blob_output_path, blob_id)
        idx_8, rho_8 = create_blob_pc_q_rank(blob_fofc, blob_center, 0.8, grid_space, blob_output_path, blob_id)
        idx_85, rho_85 = create_blob_pc_q_rank(blob_fofc, blob_center, 0.85, grid_space, blob_output_path, blob_id)
        idx_9, rho_9 = create_blob_pc_q_rank(blob_fofc, blob_center, 0.9, grid_space, blob_output_path, blob_id)
        idx_95, rho_95 = create_blob_pc_q_rank(blob_fofc, blob_center, 0.95, grid_space, blob_output_path, blob_id)

        blob_out["point_cloud_size_qRank0.5"] = len(idx_5)
        blob_out["point_cloud_size_qRank0.7"] = len(idx_7)
        blob_out["point_cloud_size_qRank0.75"] = len(idx_75)
        blob_out["point_cloud_size_qRank0.8"] = len(idx_8)
        blob_out["point_cloud_size_qRank0.85"] = len(idx_85)
        blob_out["point_cloud_size_qRank0.9"] = len(idx_9)
        blob_out["point_cloud_size_qRank0.95"] = len(idx_95)
        # create image qRankMask_5
        # make image with different values in each chanel - 0.5, 0.75 and 0.95 qRank
        # red if point only appears in the 0.5 qRank contour, yellow if appears in 0.75, and grey scale if appears in 0.95
        if len(idx_5) > 0:
            np.asarray(blob_fofc.colors)[:, :] = 0.0
            np.asarray(blob_fofc.colors)[idx_5, 0] = rho_5
            np.asarray(blob_fofc.colors)[idx_5, 1] = rho_5
            np.asarray(blob_fofc.colors)[idx_5, 2] = rho_5
        else:
            # print("Warning: No points for blob", blob_id, "in 0.95 qRank contour")
            blob_out["missing_blob"] = "ErrorPc095qRankMaskCreation"
            return blob_out
        create_blob_pc_q_rank(blob_fofc, blob_center, 'Mask_5', grid_space, blob_output_path, blob_id)
        # if len(idx_95) > 0:
        #     np.asarray(blob_fofc.colors)[:, :] = 0.0
        #     np.asarray(blob_fofc.colors)[idx_5, 0] = rho_5
        #     np.asarray(blob_fofc.colors)[idx_75, 1] = rho_75
        #     np.asarray(blob_fofc.colors)[idx_95, 2] = rho_95
        # else:
        #     # print("Warning: No points for blob", blob_id, "in 0.95 qRank contour")
        #     blob_out["missing_blob"] = "ErrorPc095qRankMaskCreation"
        #     return blob_out
        # create_blob_pc_q_rank(blob_fofc, blob_center, 'Mask_5_75_95', grid_space, blob_output_path, blob_id)
        # # create image qRankMask_5_7_9
        # if len(idx_9) > 0:
        #     np.asarray(blob_fofc.colors)[:, :] = 0.0
        #     np.asarray(blob_fofc.colors)[idx_5, 0] = rho_5
        #     np.asarray(blob_fofc.colors)[idx_7, 1] = rho_7
        #     np.asarray(blob_fofc.colors)[idx_9, 2] = rho_9
        # else:
        #     # print("Warning: No points for blob", blob_id, "in 0.95 qRank contour")
        #     blob_out["missing_blob"] = "ErrorPc09qRankMaskCreation"
        #     return blob_out
        # create_blob_pc_q_rank(blob_fofc, blob_center, 'Mask_5_7_9', grid_space, blob_output_path, blob_id)
    except Exception as e:
        # print("Error creating the blob", blob_out, "point clouds.")
        # print("Error message:", e)
        blob_out["missing_blob"] = "ErrorPcqRankCreation"
        return blob_out
    #
    blob_out["point_cloud"] = True
    d = time.time() - t1
    # print("Processed blob", blob_id, "in: %.2f s." % d)
    blob_out['time_process'] = d
    return blob_out


class EngineBlobPointCloudQRankScale(object):
    def __init__(self, blobs_grid_path, output_path, overwrite_pc, num_processors=2):
        self.blobs_grid_path = blobs_grid_path
        self.output_path = output_path
        self.overwrite_pc = overwrite_pc
        self.num_processors = num_processors
        self.verbose = (num_processors == 1)
    def call(self, blob_data_row):
        return create_blob_point_cloud_quantile_rank_scaled(blob_data_row, self.blobs_grid_path, self.output_path,
                                                            self.overwrite_pc, self.verbose)
    def run(self, blobs_retrieve):
        # pool_ligands = p_map(num_cpus=num_processors)
        blobs_out = p_map(self.call, list(blobs_retrieve.iterrows()), num_cpus=self.num_processors)
        # convert the result in a dataframe and merge with the ligands data
        blobs_out = pd.DataFrame(blobs_out)
        return blobs_out



def create_blobs_dataset_point_cloud_quantile_rank_scaled(refinement_path, entry_output_path,
                                                          num_processors, overwrite_pc):
    if num_processors > cpu_count():
        num_processors = max(cpu_count() - 1, 1)
        logging.info("Warning: The selected number of processors is greater than the available CPUs, "
              "setting it to the number of CPUs - 1 = " + str(num_processors))
    t0 = time.time()
    # check inputs
    refinement_path = Path(refinement_path)
    entry_output_path = Path(entry_output_path)
    blobs_grid_path = entry_output_path / 'model_blobs' / 'blobs_grid'
    blobs_processed_file = blobs_grid_path / 'blobs_list_processed.csv'
    logging.info("* Start the creation of the blob's point cloud images for entry " + refinement_path.name + "*")

    if not refinement_path.exists() or not refinement_path.is_dir():
        sys.exit("The provided refinement data folder do not exists.")

    if not blobs_grid_path.exists() or not blobs_grid_path.is_dir():
        logging.info("The provided refinement data folder do not have a 'model_blobs/blobs_grid' subfolder. "
                 "No blob grid creation result was found.")
        sys.exit("The provided refinement data folder do not have a 'model_blobs/blobs_grid' subfolder. "
                 "No blob grid creation result was found.")

    if not blobs_processed_file.is_file():
        logging.info("The valid list of processed blobs CSV file do not exists in the provided refinement folder "
                 "inside the 'model_blobs/blobs_grid' subfolder. The blob's grid creation probable raised an error")
        sys.exit("The valid list of processed blobs CSV file do not exists in the provided refinement folder "
                 "inside the 'model_blobs/blobs_grid' subfolder. The blob's grid creation probable raised an error")

    output_path = entry_output_path / 'model_blobs' / 'blobs_img'
    if not output_path.exists() or not output_path.is_dir():
        output_path.mkdir(parents=True)

    # read the csv with the valid blobs list, to create their point cloud image in quantile rank scale
    blobs_list = pd.read_csv(blobs_processed_file)

    # filter out the blobs with no grid created
    blobs_list = blobs_list[blobs_list.point_cloud_grid]
    n = blobs_list.shape[0]

    # run the blobs pc creation in multiprocessing
    # create the engine to create the blob pc by passing the parameters
    # blob engine
    engine_createBlobsPc = EngineBlobPointCloudQRankScale(blobs_grid_path, output_path, overwrite_pc, num_processors)
    blobs_out = engine_createBlobsPc.run(blobs_list)

    logging.info("Average time spent to process blobs: %.2f s." % blobs_out.time_process.mean())
    blobs_out = blobs_out.drop('time_process', axis=1)
    blobs_list = blobs_list.merge(blobs_out)


    skipped_blobs = blobs_list.missing_blob.str.match("ErrorBlobPcGridNoData")
    if sum(skipped_blobs) > 0:
        logging.info("- Skipped blobs due to missing point cloud grid data:")
        logging.info(blobs_list.blobID[skipped_blobs].values)

    error_blobs = blobs_list.missing_blob.str.match(".*qRankCreation|.*qRankParsing|ErrorPc095qRankMaskCreation")
    if sum(error_blobs) > 0:
        logging.info("- Error in the blobs images in quantile rank creation and/or parsing:")
        logging.info(blobs_list.blobID[error_blobs].values)
        logging.info("A total of "+str((~blobs_list["point_cloud"]).sum())+"/"+ str(n)+
              " blobs entries were missing or raised an error in the qRankMask image creation.")

    # remove failed blobs image, save csv of the created point clouds
    blobs_list = blobs_list[blobs_list["point_cloud"]]
    if blobs_list.shape[0] > 0:
        blobs_list.to_csv((output_path / 'blobs_list_processed_qRank_scale.csv').as_posix(), index=False)
    d = time.time() - t0

    logging.info("\nDONE! " + "Time elapsed: %.2f s." % d)
    return blobs_list.shape[0]



if __name__ == "__main__":
    overwrite_pc = False
    num_processors = 1
    if len(sys.argv) >= 3:
        refinement_path = sys.argv[1]
        entry_output_path = sys.argv[2]
        if len(sys.argv) >= 4:
            num_processors = int(sys.argv[3])
        if len(sys.argv) >= 5:
            overwrite_pc = (sys.argv[4].lower() == "true")

    else:
        sys.exit("Wrong number of arguments. One argument must be supplied in order to read the blobs "
                 "grid pc files, the blobs processed table file and refinement path "
                 "to extract and store the blobs point cloud in quantile rank scale: \n"
                 "  1. The path to the data folder where the entry refinement is located ('data/refinement/entryID'). "
                 "It must contain the folder 'model_blobs', inside it the subfolder 'blobs_grid' with the blob's grid "
                 "point clouds files and the list of processed blob's grids inside it in a table named "
                 "'blobs_list_processed.csv' (resulting from the mtz_to_blob_grid_pointcloud.py script). "
                 "The output point clouds will be stored in folder named 'blobs_img' inside the 'model_blobs' folder.\n"
                 "  2. The path to the output data folder where the np3 ligand result is being stored for the current"
                 "entry ('data/np3_ligand_<DATE>/entryID');\n"
                 "  3. (optional) The number of processors to use for multiprocessing parallelization "
                 "(Default to 1 - no parallelization);\n"
                 "  4. (optional) A boolean True or False indicating if the already processed blobs should be "
                 "overwritten (Default to False).\n"
                 )
    create_blobs_dataset_point_cloud_quantile_rank_scaled(refinement_path, entry_output_path,
                                                          num_processors, overwrite_pc)
