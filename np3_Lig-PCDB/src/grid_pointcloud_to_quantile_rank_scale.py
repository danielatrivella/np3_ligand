from pathlib import Path
import numpy as np
import open3d as o3d
import sys
import pandas as pd
import time
from p_tqdm import p_map
from multiprocessing import cpu_count

test_mode = False

if test_mode:
    import seaborn as sns
    import matplotlib.pyplot as plt

def plot_dist(data):
    sns.set_theme(style="darkgrid")
    sns.displot(data, height=5)
    # plt.show()
    plt.savefig('distribution_rho.png')

def read_radii_table_to_dict(radii_table_path="atomic_radii_tables.csv", radii_weight=0.65):
    # read table with atomic radii, use first row as index = atoms symbols
    radii_table = pd.read_csv(radii_table_path, index_col=0)
    # change columns names to match the resolution of each set of radii
    radii_table.columns = radii_table.columns.str.split("_").str[-1]
    # apply the weight value to the atomic radius in all resolutions
    radii_table=radii_table*radii_weight
    # return the table of atomic radii by resolution as a dictionary
    return radii_table.to_dict()

# read atomic table and apply a weight of 65%
elements_radii_w_reso = read_radii_table_to_dict(radii_table_path="atomic_radii_tables.csv", radii_weight=0.65)
expansion_radii = np.round(max(list(elements_radii_w_reso['2.2'].values())), 1)

# for testing with visual inspection color the points according to their label
elements_color_SP_test = {'0': np.array([237, 238, 192]) / 255, '1': np.array([67, 62, 14]) / 255,
                          '2': np.array([124, 144, 130]) / 255, '3': np.array([167, 162, 132]) / 255,
                          '4': np.array([208, 200, 142]) / 255, '5': np.array([242, 132, 130]) / 255,
                          '6': np.array([255, 136, 17]) / 255, '7': np.array([57, 47, 90]) / 255,
                          '8': np.array([84, 87, 124]) / 255, '9': np.array([220, 127, 155]) / 255,
                          '10': np.array([150, 173, 200]) / 255, '11': np.array([167, 194, 193]) / 255,
                          '12': np.array([183, 214, 186]) / 255, '13': np.array([215, 255, 171]) / 255,
                          '14': np.array([234, 255, 140]) / 255, '15': np.array([252, 255, 108]) / 255,
                          '16': np.array([216, 157, 106]) / 255, '17': np.array([163, 113, 91]) / 255,
                          '18': np.array([109, 69, 76]) / 255, '19': np.array([122, 86, 92]) / 255,
                          '20': np.array([59, 31, 43]) / 255, '21': np.array([219, 22, 47]) / 255,
                          '22': np.array([0, 0, 1]), '23': np.array([0, 1, 0]),
                          '24': np.array([1, 0, 0]), '25': np.array([1, 0, 1]),
                          '26': np.array([1, 1, 0])}

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
    return np.round((x - min_scale) / (max_scale - min_scale), 4)

# return the points indexes of the point cloud lig_grid_fofc connected to all x,y,z position in the lig_label_pos file
# where search_radius = grid_space * 1.42 + 0.15
def extract_ligand_point_cloud(lig_grid_fofc, lig_label_pos, grid_space):
    # print("Extract the ligand point cloud cluster using its atoms position")
    lig_grid_fofc_kdtree = o3d.geometry.KDTreeFlann(lig_grid_fofc)
    # lig_label_pos = pd.read_csv(lig_label_pos_file, skipinitialspace=True)
    #
    # apply dijkstra algorithm using the kdtree search from each atom position if necessary (if the ligand density is connected
    # # one atom will be enough). It uses a visited binary array to store the reached points
    # and a reached list to store the next points to be expanded
    search_radius = grid_space * 1.42 + 0.15  # gets the corners of the point multiplying by ~(2)^1/2 +
                                              # a gap to deal with missing points in valley regions of the electron density
    points_visited = np.zeros(len(lig_grid_fofc.points))  # 1 for reached, 2 for expanded, 0 for not seen yet
    for atom_i in range(lig_label_pos.shape[0]):
        points_reached = np.asarray(lig_grid_fofc_kdtree.search_radius_vector_3d(
            np.asarray(lig_label_pos[['x', 'y', 'z']].iloc[atom_i, :]), radius=search_radius)[1])
        # if the point was already expand continue, else remove the expanded ones and proceed for the missing one
        if (points_visited[points_reached] == 2).all():
            continue
        else:
            points_reached = points_reached[points_visited[points_reached] != 2]
        # set the initial not seen points in the neighborhood of the current atom
        points_visited[points_reached] = 1
        points_reached = list(points_reached)
        # expand each reachable point from the atom position
        while len(points_reached) > 0:
            expand_point = points_reached.pop()
            points_visited[expand_point] = 2
            neighbors = np.asarray(
                lig_grid_fofc_kdtree.search_radius_vector_3d(lig_grid_fofc.points[expand_point],
                                                             radius=search_radius)[1])
            neighbors = neighbors[points_visited[neighbors] == 0]
            points_visited[neighbors] = 1
            points_reached.extend(neighbors)
    # extract the expanded points indexes
    ligand_points_idx = np.where(points_visited == 2)[0]
    return ligand_points_idx

# extract the ligand point cloud for a given quantile rank contour value and
# return the points indexes in the given lig_pc_pruned
def create_lig_pc_q_rank(lig_pc_mask, points_label, q_rank_contour, lig_label_pos, grid_space,
                         lig_output_path, lig_name):
    if q_rank_contour is not None:
        if q_rank_contour in ["Mask_5", "Mask_5_75_95", "Mask_5_7_9"]:
            file_label = q_rank_contour
        else:  # apply a contour
            file_label = str(q_rank_contour)
            lig_pc_qrank, pc_qrank_idx = extract_pc_gt_value(lig_pc_mask, q_rank_contour)
            # if selected less points, extract ligand pc again to remove isolated noise
            if len(pc_qrank_idx) < len(lig_pc_mask.points):
                # extract ligand point cloud idx
                lig_sel = extract_ligand_point_cloud(lig_pc_qrank, lig_label_pos, grid_space)
                lig_pc_mask = lig_pc_qrank.select_by_index(lig_sel)
                points_label = points_label[pc_qrank_idx[lig_sel]]
                pc_qrank_idx = pc_qrank_idx[lig_sel]
    else:
        file_label = "Mask"
#        q_rank_contour = np.min(np.asarray(lig_pc_mask.colors)[:, 0])
    # min_max scale image if it is not the image with different scales in each chanel
    if file_label not in ["Mask_5", "Mask_5_75_95", "Mask_5_7_9", "Mask"]:
        np.asarray(lig_pc_mask.colors)[:, :] = min_max_scale(np.asarray(lig_pc_mask.colors)[:, :],
                                                             q_rank_contour,
                                                             1.0)
    if test_mode:
        o3d.visualization.draw_geometries([lig_pc_mask])
        plot_dist(np.asarray(lig_pc_mask.colors)[:, 0])
        if file_label in ["Mask_5", "Mask_5_75_95", "Mask_5_7_9","Mask"]:
            plot_dist(np.asarray(lig_pc_mask.colors)[:, 1])
            plot_dist(np.asarray(lig_pc_mask.colors)[:, 2])
    # save the ligand point cloud and labels if their sizes match
    if len(lig_pc_mask.points) != len(points_label):
        print("Wrong number of point labels. Not matching with the total number of points")
        sys.exit(1)
    o3d.io.write_point_cloud(
        (lig_output_path / (lig_name + "_lig_point_cloud_fofc_qRank"+file_label+".xyzrgb")).as_posix(),
        lig_pc_mask)
    if file_label not in ["Mask_5", "Mask_5_75_95", "Mask_5_7_9"]:  # the labels for this image is the same of the mask labels
        np.savetxt((lig_output_path / (lig_name + "_lig_pc_labels_qRank"+file_label+".txt")).as_posix(),
                   points_label, fmt="%s")
    # return the indexes of the contour and their scaled intensity values
    if file_label not in ["Mask", "Mask_5", "Mask_5_75_95", "Mask_5_7_9"]:
        return pc_qrank_idx, np.asarray(lig_pc_mask.colors)[:, 0]
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

# extract the ligand point cloud for a given quantile rank contour value and return it
def extract_lig_pc_Q_rank_idx(lig_grid, q_rank_contour, lig_label_pos, grid_space):
    # print("Extract the ligand point cloud for",q_rank_contour, "quantile rank")
    # select values above the contour
    lig_q_rank, lig_q_rank_sel_idx = extract_pc_gt_value(lig_grid, q_rank_contour)
    if len(lig_q_rank.points) > 0:
        # extract ligand point cloud
        lig_sel = extract_ligand_point_cloud(lig_q_rank, lig_label_pos, grid_space)
        if test_mode:
            o3d.visualization.draw_geometries([lig_q_rank.select_by_index(lig_sel)])
    else:
        return None
    # return the extracted ligand pc indexes in the given contour
    return lig_q_rank_sel_idx[lig_sel]
    # return lig_q_rank.select_by_index(lig_sel)

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

# def quantile_rank_scale(p_grid_values, bin_size=0.05):
#     n_grid = p_grid_values.size
#     p_min, p_max = min_max(p_grid_values)
#     p_min = p_min.round(3) - (0.01 if p_min.round(3) > p_min else 0.0)
#     p_max = p_max.round(3) + (0.01 if p_max.round(3) < p_max else 0.0)
#     J = np.ceil((p_max - p_min)/bin_size).astype(np.int64)+1  # number of bins in the p values interval
#     n = np.zeros(J, np.int64)  # nj counter
#     p = lambda j : (p_min + bin_size * j).round(4)  # p values of each bin
#     # compute nj: count the number of points in each bin such that pj-1 <= p(n) < pj and j in 1:J
#     for j in range(1,J+1):
#         n[j-1] = ((p_grid_values >= p(j-1)) & (p_grid_values < p(j))).sum()
#     # compute the frequencies
#     v = n / n_grid
#     # compute the quantile rank q = N(pj)/Ngrid = sum_i<=j vi, where N(pj)=number of points with p(n) < pj
#     q = v.cumsum()
#     # replace p(n) at each point in the grid with the quantile rank q, interpolate for intermediate values
#     bin_interval = lambda p_n: [np.floor((p_n - p_min).round(4)/bin_size).astype(np.int64), np.ceil((p_n - p_min).round(4)/bin_size).astype(np.int)]
#     q_grid_values = np.zeros(n_grid)
#     for i in range(n_grid):
#         bin_j = bin_interval(p_grid_values[i])
#         if bin_j[0] == bin_j[1]:  # exact the bin value
#             q_grid_values[i] = q[bin_j[0]]
#         else:  # interpolate
#             q_grid_values[i] = q[bin_j[0]] + (q[bin_j[1]]-q[bin_j[0]]) * (p_grid_values[i] - p(bin_j[0])) / \
#                                (p(bin_j[1]) - p(bin_j[0]))
#     # return density values rank scaled in the ligand grid
#     return q_grid_values

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


def create_ligand_point_cloud_quantile_rank_scaled(lig_data_row, ligs_grid_path, output_path, db_ligxyz_path,
                                                   expansion_radii, atom_dist_filter, overwrite_pc):
    # i = lig_data_row[0]
    lig_data = lig_data_row[1]
    ligand_out = pd.Series({'ligID': lig_data['ligID']})
    ligand_out["point_cloud"] = False
    ligand_out["point_cloud_size"] = -1  # 0.95 quantile rank mask expanded
    ligand_out["point_cloud_size_qRank0.5"] = -1
    ligand_out["point_cloud_size_qRank0.7"] = -1
    ligand_out["point_cloud_size_qRank0.75"] = -1
    ligand_out["point_cloud_size_qRank0.8"] = -1
    ligand_out["point_cloud_size_qRank0.85"] = -1
    ligand_out["point_cloud_size_qRank0.9"] = -1
    ligand_out["point_cloud_size_qRank0.95"] = -1
    ligand_out["missing_ligand"] = "False"
    ligand_out['time_process'] = None
    # create the grid and point cloud for each valid ligand
    # validate lig output path
    entry_name = lig_data['entry']
    lig_name = lig_data['ligID']
    grid_space = lig_data["grid_space"]
    lig_grid_path = ligs_grid_path / entry_name
    lig_output_path = output_path / entry_name

    if not lig_grid_path.exists():
        # print("Warning: The PDB entry ", entry_name,
        #       "grid pointclouds were not created. All its ligands will be skipped.")
        ligand_out["missing_ligand"] = "ErrorPDBPcData"
        return ligand_out
    elif not (lig_grid_path / (lig_name + "_grid_point_cloud_fofc.xyzrgb")).exists():
        # print("Warning: The ligand ", lig_name,
        #       "grid pointcloud was not created. Skipping to the next ligand entry.")
        ligand_out["missing_ligand"] = "ErrorPcGridCreation"
        return ligand_out

    # for each ligand in the current entry, scale the electron density values to the quantile rank,
    # extract a mask with 0.95 contour of the ligand pc with expansion and
    # create the point clouds in others quantiles contours inside the mask

    # if the ligand was already processed and overwrite is false, only get the pcs sizes and skip the ligand processing
    if (lig_output_path / (lig_name + "_lig_point_cloud_fofc_qRankMask.xyzrgb")).exists() and not overwrite_pc:
        # print("* Already processed ligand", lig_name, i + 1, "/", n, ". Skipping to next entry. *")
        ligand_out["point_cloud"] = True
        ligand_out["point_cloud_size"] = rawcount(
            (lig_output_path / (lig_name + "_lig_point_cloud_fofc_qRankMask.xyzrgb")))
        for qRank_contour in ["qRank0.95", "qRank0.9", "qRank0.85", "qRank0.8", "qRank0.75", "qRank0.7", "qRank0.5"]:
            ligand_out["point_cloud_size_"+qRank_contour] = rawcount(
                (lig_output_path / (lig_name + "_lig_point_cloud_fofc_"+qRank_contour+".xyzrgb")))
            if ligand_out["point_cloud_size_"+qRank_contour] == 0:
                # print("Warning: No points for ligand", lig_name, "in quantile rank contour")
                ligand_out["missing_ligand"] = "ErrorPc"+qRank_contour+"Parsing"
        #
        return ligand_out

    # create the ligand output path if it does not exists
    #if not lig_output_path.exists() or not lig_output_path.is_dir():
    lig_output_path.mkdir(parents=True, exist_ok=True)
    # round the ligand pdb entry resolution
    reso = str(round(float(lig_data['Resolution']),1))
    # print("\n* Start processing ligand entry", lig_name, "-", i + 1, "/", n, "*")
    t1 = time.time()
    # get the ligand label file path
    lig_label_pos_file = db_ligxyz_path / (lig_name + '_class.xyz')

    try:
        # read grid point cloud fofc and scale the density values using the quantile rank
        lig_grid = o3d.io.read_point_cloud(
            (lig_grid_path / (lig_name + "_grid_point_cloud_fofc.xyzrgb")).as_posix())
        # rho_Q_rank = quantile_rank_scale(p_grid_values=np.asarray(lig_grid.colors)[:, 0], bin_size=0.005)
        rho_Q_rank = quantile_rank_scale_fast(p_grid_values=np.asarray(lig_grid.colors)[:, 0])
        # print("Number of points in the ligand grid", len(lig_grid.points))
        np.asarray(lig_grid.colors)[:, 0] = rho_Q_rank
        np.asarray(lig_grid.colors)[:, 1] = rho_Q_rank
        np.asarray(lig_grid.colors)[:, 2] = rho_Q_rank
        del rho_Q_rank

        # read ligand coordinate file and labels
        lig_label_pos = pd.read_csv(lig_label_pos_file, skipinitialspace=True)
        # extract the ligand point cloud using a mask with 0.95 quantile rank expanded to remove background noise
        # before applying other contour levels
        lig_fofc_idx = extract_lig_pc_Q_rank_idx(lig_grid, 0.95, lig_label_pos, grid_space)
        # check if the number of points at 0.95 contour is below 20% of the number of points of the smaller atom sphere
        # if yes skip this entry
        if len(lig_fofc_idx) < (4 / 3 * np.pi * np.power(min(elements_radii_w_reso['1.5'].values()), 3) / np.power(grid_space, 3)):
            raise Exception
        lig_fofc = extract_pc_neighborhood(lig_grid, lig_fofc_idx, expansion_radii)
        #print("aa")
        # print("Number of points in the ligand point cloud 0.95 quantile rank mask", len(lig_fofc.points))
        # draw the geometries
        if test_mode:
            # print("Plot rho grid points distribution and the ligand pc in 0.95 quantile rank mask expanded")
            plot_dist(np.asarray(lig_grid.colors)[:, 0])
            o3d.visualization.draw_geometries([lig_fofc])
        del lig_grid, lig_fofc_idx
    except Exception as e:
        # print("Error message:", e)
        ligand_out["missing_ligand"] = "ErrorPc095qRankMaskCreation"
        return ligand_out

    # print("- Label the ligand's points using the distance to the ligand's atoms")
    try:
        # label points according to their distance to the closest atom and the respective atom radii
        lig_kdtree = o3d.geometry.KDTreeFlann(lig_fofc)
        # store the distance to the atom used to color each point
        points_labeled_dist = np.full(len(lig_fofc.points), -1.0)
        points_label = np.full(len(lig_fofc.points), "-1")
        # expand mask in 10% to deal with atoms contours that are greater than the atom radii
        # expansion_radii = expansion_radii*1.1
        # if test mode save original density values
        if test_mode:
            p_q_rank = np.array(lig_fofc.colors, copy=True)
            # plot_dist(p_q_rank[:, 0])
        for atom_i in range(lig_label_pos.shape[0]):
            # select points that are within the atom radii from the current atom position+expansion radii
            neighbors_points = list(
                lig_kdtree.search_radius_vector_3d(np.array(lig_label_pos[['x', 'y', 'z']].iloc[atom_i, :]),
                                                   elements_radii_w_reso[reso][lig_label_pos.symbol[atom_i]]+
                                                   expansion_radii))
            neighbors_points[2] = np.asarray(neighbors_points[2])
            # assign distance to points if they are not labeled yet or if the distance to this atom is smaller
            points_selected_label = ((points_labeled_dist[neighbors_points[1]] == -1) |
                                     (points_labeled_dist[neighbors_points[1]] > neighbors_points[2]))
            idx_selected_pts = np.asarray(neighbors_points[1])[points_selected_label]
            # update minimum distance to each reached point
            points_labeled_dist[idx_selected_pts] = neighbors_points[2][points_selected_label]
            # select atoms that are within the atom radii to be labeled with the atom label
            idx_selected_pts = idx_selected_pts[neighbors_points[2][points_selected_label] <=
                                                np.power(elements_radii_w_reso[reso][lig_label_pos.symbol[atom_i]], 2)]
            points_label[idx_selected_pts] = lig_label_pos.labels[atom_i]
            # only color in test mode
            if test_mode:
                np.asarray(lig_fofc.colors)[idx_selected_pts, :] = elements_color_SP_test[
                    str(lig_label_pos.labels[atom_i])]

        # remove too far away points, depending on the parameter atom_dist_filter
        if atom_dist_filter:
            lig_fofc = lig_fofc.select_by_index(np.where(points_labeled_dist > -1)[0])
            points_label = points_label[points_labeled_dist > -1]

        if test_mode:
            print("draw colored labels")
            o3d.visualization.draw_geometries([lig_fofc])
            # restore original density values if in test mode
            np.asarray(lig_fofc.colors)[:, :] = p_q_rank[points_labeled_dist > -1]
    except Exception as e:
        #print("Error labeling the ligand", lig_name, "point cloud.")
        #print("Error message:", e)
        ligand_out["missing_ligand"] = "ErrorPcLabeling"
        # error_ligands.append(lig_name+'_label')
        return ligand_out
    if len(lig_fofc.points) != len(points_label):
        print("Wrong number of point labels. Not matching with the total number of points")
        sys.exit(1)

    # # save the ligand points labels
    # np.savetxt((lig_output_path / (lig_name + "_lig_pc_labels_qRankMask.txt")).as_posix(),
    #            points_label, fmt="%s")
    # print("Number of point labels", len(points_label))
    ligand_out["point_cloud"] = True
    ligand_out["point_cloud_size"] = len(lig_fofc.points)
    #
    # create the ligand point cloud in quantile rank scaled in the 0.95 quantile mask expanded, and
    # with 0.7, 0.8 and 0.9 contour inside the mask
    try:
        create_lig_pc_q_rank(lig_fofc, points_label, None, lig_label_pos, grid_space, lig_output_path, lig_name)
        idx_5, rho_5 = create_lig_pc_q_rank(lig_fofc, points_label, 0.5, lig_label_pos, grid_space, lig_output_path, lig_name)
        idx_7, rho_7 = create_lig_pc_q_rank(lig_fofc, points_label, 0.7, lig_label_pos, grid_space, lig_output_path, lig_name)
        idx_75, rho_75 = create_lig_pc_q_rank(lig_fofc, points_label, 0.75, lig_label_pos, grid_space, lig_output_path, lig_name)
        idx_8, rho_8 = create_lig_pc_q_rank(lig_fofc, points_label, 0.8, lig_label_pos, grid_space, lig_output_path,
                                            lig_name)
        idx_85, rho_85 = create_lig_pc_q_rank(lig_fofc, points_label, 0.85, lig_label_pos, grid_space, lig_output_path,
                                              lig_name)
        idx_9, rho_9 = create_lig_pc_q_rank(lig_fofc, points_label, 0.9, lig_label_pos, grid_space, lig_output_path, lig_name)
        idx_95, rho_95 = create_lig_pc_q_rank(lig_fofc, points_label, 0.95, lig_label_pos, grid_space, lig_output_path, lig_name)

        ligand_out["point_cloud_size_qRank0.5"] = len(idx_5)
        ligand_out["point_cloud_size_qRank0.7"] = len(idx_7)
        ligand_out["point_cloud_size_qRank0.75"] = len(idx_75)
        ligand_out["point_cloud_size_qRank0.8"] = len(idx_8)
        ligand_out["point_cloud_size_qRank0.85"] = len(idx_85)
        ligand_out["point_cloud_size_qRank0.9"] = len(idx_9)
        ligand_out["point_cloud_size_qRank0.95"] = len(idx_95)
        # create image qRankMask_5
        if len(idx_5) > 0:
            np.asarray(lig_fofc.colors)[:, :] = 0.0
            np.asarray(lig_fofc.colors)[idx_5, 0] = rho_5
            np.asarray(lig_fofc.colors)[idx_5, 1] = rho_5
            np.asarray(lig_fofc.colors)[idx_5, 2] = rho_5
        else:
            # print("Warning: No points for ligand", lig_name, "in 0.5 qRank contour")
            ligand_out["missing_ligand"] = "ErrorPc05qRankMaskCreation"
            return ligand_out
        create_lig_pc_q_rank(lig_fofc, points_label, 'Mask_5', lig_label_pos, grid_space,
                             lig_output_path, lig_name)
        # create image qRankMask_5_75_95
        # make image with different values in each chanel - 0.5, 0.75 and 0.95 qRank
        # red if point only appears in the 0.5 qRank contour, yellow if appears in 0.75, and grey scale if appears in 0.95
#        if len(idx_95) > 0:
#            np.asarray(lig_fofc.colors)[:, :] = 0.0
#            np.asarray(lig_fofc.colors)[idx_5, 0] = rho_5
#            np.asarray(lig_fofc.colors)[idx_75, 1] = rho_75
#            np.asarray(lig_fofc.colors)[idx_95, 2] = rho_95
#        else:
#            # print("Warning: No points for ligand", lig_name, "in 0.95 qRank contour")
#            ligand_out["missing_ligand"] = "ErrorPc095qRankMaskCreation"
#            return ligand_out
#        create_lig_pc_q_rank(lig_fofc, points_label, 'Mask_5_75_95', lig_label_pos, grid_space,
#                             lig_output_path, lig_name)
#        if len(idx_9) > 0:
#            np.asarray(lig_fofc.colors)[:, :] = 0.0
#            np.asarray(lig_fofc.colors)[idx_5, 0] = rho_5
#            np.asarray(lig_fofc.colors)[idx_7, 1] = rho_7
#            np.asarray(lig_fofc.colors)[idx_9, 2] = rho_9
#        else:
#            # print("Warning: No points for ligand", lig_name, "in 0.95 qRank contour")
#            ligand_out["missing_ligand"] = "ErrorPc095qRankMaskCreation"
#            return ligand_out
#        create_lig_pc_q_rank(lig_fofc, points_label, 'Mask_5_7_9', lig_label_pos, grid_space,
#                             lig_output_path, lig_name)
    except Exception as e:
        # print("Error creating the ligand", lig_name, "point clouds.")
        # print("Error message:", e)
        ligand_out["missing_ligand"] = "ErrorPcqRankCreation"
        # error_ligands.append(lig_name+'_label')
        return ligand_out

    d = time.time() - t1
    # print("Processed ligand", lig_name, "in: %.2f s." % d)
    ligand_out['time_process'] = d
    return ligand_out


class EngineLigandPointCloudQRankScale(object):
    def __init__(self, ligs_grid_path, output_path, db_ligxyz_path, atom_dist_filter, overwrite_pc):
        self.ligs_grid_path = ligs_grid_path
        self.output_path = output_path
        self.db_ligxyz_path = db_ligxyz_path
        self.atom_dist_filter = atom_dist_filter
        self.overwrite_pc = overwrite_pc

    def call(self, lig_data_row):
        return create_ligand_point_cloud_quantile_rank_scaled(lig_data_row, self.ligs_grid_path, self.output_path,
                                                              self.db_ligxyz_path, expansion_radii,
                                                              self.atom_dist_filter,
                                                              self.overwrite_pc)
    def run(self, ligs_retrieve, num_processors):
        # pool_ligands = p_map(num_cpus=num_processors)
        ligands_out = p_map(self.call, list(ligs_retrieve.iterrows()), num_cpus=num_processors)
        # convert the result in a dataframe and merge with the ligands data
        ligands_out = pd.DataFrame(ligands_out)
        return ligands_out


def create_ligands_dataset_point_cloud_quantile_rank_scaled(db_ligxyz_path, ligs_grid_path, output_path,
                                                            atom_dist_filter, num_processors = 2, overwrite_pc=False,
                                                            i_start=0, n=0):
    if n <= i_start and n > 0:
        sys.exit("Wrong range: The provided stop row is not greater than the start row.")
    if num_processors > cpu_count():
        num_processors = max(cpu_count() - 1, 1)
        print("Warning: The selected number of processors is greater than the available CPUs, setting it to the number of CPUs - 1 = ",
              num_processors)
    t0 = time.time()
    # check inputs
    db_ligxyz_path = Path(db_ligxyz_path)
    valid_ligs_file = db_ligxyz_path / (db_ligxyz_path.name.replace("xyz_", "") + '_box_class_freq.csv')
    valid_ligs_pc_file = db_ligxyz_path / (db_ligxyz_path.name.replace("xyz_", "") + '_box_pc.csv')

    if not db_ligxyz_path.exists() or not db_ligxyz_path.is_dir():
        sys.exit("The provided ligand xyz data folder do not exists.")
    if not Path(valid_ligs_file).is_file() or not Path(valid_ligs_pc_file).is_file():
        sys.exit("The valid ligands list CSV file do not exists in the provided ligand xyz data folder.")

    ligs_grid_path = Path(ligs_grid_path)
    if not ligs_grid_path.exists() or not ligs_grid_path.is_dir():
        sys.exit("The provided ligand point cloud grids data folder do not exists.")

    output_path = Path(output_path)
    if not output_path.exists() or not output_path.is_dir():
        output_path.mkdir(parents=True)

    # read csv with the valid ligands list to encode
    ligs_retrieve = pd.read_csv(valid_ligs_file, na_values = ['null', 'N/A'], keep_default_na = False,
                                usecols = ['ligID', 'ligCode', 'entry', 'Resolution', 'filter_quality', 'x', 'y', 'z',
                                           'x_bound', 'y_bound', 'z_bound']) # do not interpret sodium NA as nan
    ligs_retrieve_grid_space = pd.read_csv(valid_ligs_pc_file, na_values=['null', 'N/A'], keep_default_na=False,
                                usecols=['ligID', 'grid_space'])
    ligs_retrieve = ligs_retrieve.merge(ligs_retrieve_grid_space)
    del ligs_retrieve_grid_space
    # filter out the low quality ligands
    ligs_retrieve = ligs_retrieve[ligs_retrieve.filter_quality]
    if n == 0 or n > ligs_retrieve.shape[0]:
        n = ligs_retrieve.shape[0]

    # run the ligand pc creation in multiprocessing
    # create the engine to create the ligand pc passing the parameters
    engine_createLigsPc = EngineLigandPointCloudQRankScale(ligs_grid_path, output_path, db_ligxyz_path, atom_dist_filter,
                                                           overwrite_pc)
    ligands_out = engine_createLigsPc.run(ligs_retrieve.iloc[i_start:n, :], num_processors)

    print("Average time spent to process ligands: %.2f s." % ligands_out.time_process.mean())
    ligands_out = ligands_out.drop('time_process', axis=1)
    ligs_retrieve = ligs_retrieve.merge(ligands_out)

    # print missing entries and skipped ligands and error ligands
    entries_missing = ligs_retrieve.missing_ligand.str.match("ErrorPDBPcData")
    if sum(entries_missing) > 0:
        print("- Missing PDB entries:")
        print(ligs_retrieve.entry[entries_missing].unique())

    skipped_ligands = ligs_retrieve.missing_ligand.str.match("ErrorPDBPcData|ErrorPcGridCreation")
    if sum(skipped_ligands) > 0:
        print("- Skipped ligands due to missing point cloud grid data:")
        print(ligs_retrieve.ligID[entries_missing].values)

    error_ligands = ligs_retrieve.missing_ligand.str.match("ErrorPcLabeling|.*qRankCreation|.*qRankParsing|ErrorPc095qRankMaskCreation")
    if sum(error_ligands) > 0:
        print("- Error ligands in labeling, point cloud in quantile rank creation and/or parsing:")
        print(ligs_retrieve.ligID[error_ligands].values)

    if sum(entries_missing) > 0:
        print("A total of "+str(len(ligs_retrieve.entry[entries_missing].unique()))+"/",
              str(len(ligs_retrieve[i_start:n].entry.unique())),
              " PDB entries point clouds data was missing.")

    if sum(skipped_ligands) > 0:
        print("A total of "+str(sum(skipped_ligands))+"/", str(n-i_start),
              " ligands entries were skipped due to missing PDB entries point clouds data.")

    if sum(error_ligands) > 0:
        print("A total of "+str((~ligs_retrieve["point_cloud"]).sum())+"/", str(n-i_start),
              " ligands entries were missing or raised an error before the qRankMask image creation.")

    # remove failed ligands, save csv of the created point clouds and print number of pdb and ligand errors
    ligs_retrieve = ligs_retrieve[ligs_retrieve["point_cloud"]]
    ligs_retrieve.to_csv((db_ligxyz_path / valid_ligs_file.name.replace(".csv", "_qRank_scale.csv")).as_posix(),
                         index=False)
    d = time.time() - t0

    print("\nDONE!", "Time elapsed: %.2f s." % d)


if __name__ == "__main__":
    # read the ligands csv with respective pdb entries, x, y, z and data
    atom_dist_filter = False # hardcoded to be False, this filter is not used anymore -> A boolean True or False indicating if the filter to remove points too far away from an atom should be applied in the ligand point clouds creation. (Default to False, not applying the filter in the obtained cloud)
    overwrite_pc = False
    num_processors=2
    i_start = n = 0
    if len(sys.argv) >= 4:
        db_ligxyz_path = sys.argv[1]
        ligs_grid_path = sys.argv[2]
        output_path = sys.argv[3]
        if len(sys.argv) >= 5:
            num_processors = int(sys.argv[4])
        # if len(sys.argv) >= 6:
        #     atom_dist_filter = (sys.argv[5].lower() == "true")
        if len(sys.argv) >= 6:
            overwrite_pc = (sys.argv[5].lower() == "true")
        if len(sys.argv) >= 7:
            i_start = int(sys.argv[6])
        if len(sys.argv) >= 8:
            n = int(sys.argv[7])
    else:
        sys.exit("Wrong number of arguments. Three parameters must be supplied in order to read the ligands "
                 "mtz file, xyz file and output path to create the final images of the ligands in quantile rank scale "
                 "and store to point clouds. The output folder will be a Lig-PCDB database.\nList of parameters: \n"
                 "  1. xyz_labels_path: The path to the data folder called 'xyz_<valid ligand list csv name>' where the ligands .xyz files with "
                 "their labels are located. It must also contain the CSV file with the valid ligands list that had the "
                 "ligand grid image successfully created. This file is named as '<valid ligand list csv name>_box_pc.csv' and is "
                 "expected to be the output of the mtz_to_grid_pointcloud.py script. "
                 "Mandatory columns = 'ligID', 'ligCode', 'entry';\n"
                 "  2. output_grid_path: The path to the folder where the ligands grid image in point clouds are stored ('data/lig_point_clouds_grids');\n"
                 "  3. output_ligPCDB_path: The path to the output folder where the point clouds of the final images of "
                 "the ligands in quantile rank scale will be stored ('data/lig_pcdb' or other);\n"
                 "  4. num_parallel: (optional) The number of processors to use for multiprocessing parallelization (Default to 2);\n"
                 "  5. overwrite: (optional) A boolean True or False indicating if the already processed ligands should be "
                 "overwritten (Default to False);\n"
                 "  6. row_start: (optional) The number of the row of the '<valid ligand list csv name>_box_pc.csv' file where the script should start. "
                 "Skip to the given row or, if missing, start from the beginning;\n"
                 "  7. row_end: (optional) The number of the row of the '<valid ligand list csv name>_box_pc.csv' file where the script should stop. "
                 "Stop in the given row or, if missing, equal to zero or greater than the number of rows, stop in the last row.\n"
                 )
    create_ligands_dataset_point_cloud_quantile_rank_scaled(db_ligxyz_path, ligs_grid_path, output_path, atom_dist_filter,
                                                            num_processors, overwrite_pc, i_start, n)
