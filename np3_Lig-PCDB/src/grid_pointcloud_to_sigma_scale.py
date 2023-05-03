from pathlib import Path
import numpy as np
import open3d as o3d
import sys
import pandas as pd
import time
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

test_mode = False

if test_mode:
    import seaborn as sns
    import matplotlib.pyplot as plt

def plot_dist(data):
    sns.set_theme(style="darkgrid")
    sns.displot(data, height=5)
    # plt.show()
    plt.savefig('distribution_rho.png')

# the Van der Waals Radii of Elements from  S. S. Batsanov 2001
# elements_radii = {'B': 1.8, 'C': 1.7,
#                   'N': 1.6, 'P': 1.95,
#                   'O': 1.55, 'S': 1.8, 'Se': 1.9,
#                   'Cl': 1.8, 'F': 1.5, 'Br': 1.9, 'I': 2.1}
# the experimental Van der Waals Radii averaged from 1.5A to 1.8A rounded in 2 decimals from XGen 2020
# - B received S and Se received Br radii
elements_radii = {'B': 1.4, 'C': 1.46,
                  'N': 1.44, 'P': 1.4,
                  'O': 1.42, 'S': 1.4, 'Se': 1.37,
                  'Cl': 1.4, 'F': 1.4, 'Br': 1.37, 'I': 1.37}
expansion_radii = np.round(max(elements_radii.values()), 1)

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

def robust_sigma_scale_clipping(x, median_I, q25th, q75th, clipping_value=4.0):
    x_sigma_scale = (np.asarray(x) - median_I) / (q75th - q25th)
    # clipping +- clipping value
    clipping_logbase = clipping_value-np.log(clipping_value)
    x_sigma_scale[x_sigma_scale > clipping_value] = clipping_logbase + np.log(x_sigma_scale[x_sigma_scale > clipping_value])
    x_sigma_scale[x_sigma_scale < -clipping_value] = -(clipping_logbase + np.log(-1*x_sigma_scale[x_sigma_scale < -clipping_value]))
    return x_sigma_scale

def min_max_scale(x, min_scale, max_scale):
    return (x - min_scale) / (max_scale - min_scale)

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

# extract the ligand point cloud for a given sigma contour value and return the points indexes in the given lig_pc_pruned
def create_lig_pc_sigma(p_sigma, lig_pc, points_label, sigma_contour, lig_label_pos, grid_space, lig_output_path, lig_name):
    # print("Creating ligand point cloud for",sigma_contour, "sigma contour and scaling with the min and max of the resulting image")
    sigma_contour_idx = (p_sigma[:, 0] > sigma_contour)
    sigma_contour_idx_where = np.where(sigma_contour_idx)[0]
    if len(sigma_contour_idx_where) > 0:
        lig_sigma_sel = lig_pc.select_by_index(sigma_contour_idx_where)
        # extract ligand point cloud
        lig_sel = extract_ligand_point_cloud(lig_sigma_sel, lig_label_pos, grid_space)
        sigma_contour_idx[:] = False
        sigma_contour_idx[sigma_contour_idx_where[lig_sel]] = True
        if not sigma_contour_idx.any():  # no points extracted
            return sigma_contour_idx
        # min_max scale image
        lig_sigma_sel = lig_sigma_sel.select_by_index(lig_sel)
        np.asarray(lig_sigma_sel.colors)[:, :] = min_max_scale(p_sigma[sigma_contour_idx, :],
                                                               sigma_contour,
                                                               np.max(p_sigma[sigma_contour_idx, 0]))
        if test_mode:
            o3d.visualization.draw_geometries([lig_sigma_sel])
        # save the ligand point cloud and labels
        o3d.io.write_point_cloud(
            (lig_output_path / (lig_name + "_lig_point_cloud_fofc_"+str(sigma_contour)+"sigma.xyzrgb")).as_posix(),
            lig_sigma_sel)
        np.savetxt((lig_output_path / (lig_name + "_lig_pc_labels_"+str(sigma_contour)+"sigma.txt")).as_posix(),
                   points_label[sigma_contour_idx], fmt="%s")
        if len(lig_sel) != len(points_label[sigma_contour_idx]):
            print("Wrong number of point labels. Not matching with the total number of points")
            sys.exit(1)
    return sigma_contour_idx

def extract_pc_gt_value(pc, value):
    # select values above the contour
    pc_sel_idx = np.where(np.asarray(pc.colors)[:, 0] > value)[0]
    pc_sel = pc.select_by_index(pc_sel_idx)
    return pc_sel, pc_sel_idx

# extract the ligand point cloud for a given sigma contour value and return it
def extract_lig_pc_sigma_idx(lig_grid, median_I, q25th_I, q75th_I, sigma_contour, lig_label_pos, grid_space):
    # print("Extract the ligand point cloud for",sigma_contour, "sigma contour")
    # select values above the contour
    lig_sigma, lig_sigma_sel_idx = extract_pc_gt_value(lig_grid, (median_I + (q75th_I-q25th_I) * sigma_contour))
    if len(lig_sigma.points) > 0:
        # extract ligand point cloud
        lig_sel = extract_ligand_point_cloud(lig_sigma, lig_label_pos, grid_space)
        if test_mode:
            o3d.visualization.draw_geometries([lig_sigma.select_by_index(lig_sel)])
    else:
        return None
    # return the extracted ligand pc indexes in the given contour
    return lig_sigma_sel_idx[lig_sel]

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


def create_ligand_point_cloud_sigma_scaled(lig_data_row, ligs_grid_path, output_path, db_ligxyz_path,
                                           grid_space, overwrite_pc, pbar):
    # i = lig_data_row[0]
    lig_data = lig_data_row[1]
    ligand_out = pd.Series({'ligID': lig_data['ligID']})
    ligand_out["point_cloud"] = False
    ligand_out["point_cloud_size"] = -1  # 2sigmaMask
    ligand_out["point_cloud_size_05sigma"] = -1
    ligand_out["point_cloud_size_1sigma"] = -1
    ligand_out["point_cloud_size_2sigma"] = -1
    ligand_out["point_cloud_size_3sigma"] = -1
    ligand_out["missing_ligand"] = "False"
    ligand_out['time_process'] = None
    # create the grid and point cloud for each valid ligand
    # for i in range(i_start, n):
    # validate lig output path
    entry_name = lig_data["entry"]
    lig_name = lig_data['ligID']
    lig_grid_path = ligs_grid_path / entry_name
    lig_output_path = output_path / entry_name

    if not lig_grid_path.exists():
        # print("Warning: The PDB entry ", entry_name,
        #       "grid pointclouds were not created. All its ligands will be skipped.")
        # if entries_missing.count(entry_name) == 0:
        #     entries_missing.append(entry_name)
        ligand_out["missing_ligand"] = "ErrorPDBPcData"
        # skipped_ligands.append(ligs_retrieve.loc[i, 'ligID']+"_missingPcFolder")
        pbar.update(1)
        return ligand_out
    elif not (lig_grid_path / (lig_name + "_grid_point_cloud_fofc.xyzrgb")).exists():
        # print("Warning: The ligand ", lig_name,
        #       "grid pointcloud was not created. Skipping to the next ligand entry.")
        ligand_out["missing_ligand"] = "ErrorPcGridCreation"
        # skipped_ligands.append(ligs_retrieve.loc[i, 'ligID'] + "_missingPcGrid")
        pbar.update(1)
        return ligand_out

    # for each ligand in the current entry, extract the 2 sigma scale Mask with 2A expasion from the grid point cloud and create
    # the sigma scaled point clouds with different contours inside this mask

    # if the ligand was already processed and overwrite is false, only get the pcs sizes and skip the ligand processing
    if (lig_output_path / (lig_name + "_lig_point_cloud_fofc_1_2sigmaMask.xyzrgb")).exists() and not overwrite_pc:
        # print("* Already processed ligand", lig_name, i + 1, "/", n, ". Skipping to next entry. *")
        ligand_out["point_cloud"] = True
        ligand_out["point_cloud_size"] = rawcount(
            (lig_output_path / (lig_name + "_lig_point_cloud_fofc_2sigmaMask.xyzrgb")))
        ligand_out["point_cloud_size_05sigma"] = rawcount(
            (lig_output_path / (lig_name + "_lig_point_cloud_fofc_0.5sigma.xyzrgb")))
        ligand_out["point_cloud_size_1sigma"] = rawcount(
            (lig_output_path / (lig_name + "_lig_point_cloud_fofc_1.0sigma.xyzrgb")))
        if ligand_out["point_cloud_size_1sigma"] == 0:
            # print("Warning: No points for ligand", lig_name, "in 1 sigma contour")
            ligand_out["missing_ligand"] = "ErrorPc1sigmaParsing"
            # error_ligands.append(lig_name + '_sigma_1')
        ligand_out["point_cloud_size_2sigma"] = rawcount(
            (lig_output_path / (lig_name + "_lig_point_cloud_fofc_2.0sigma.xyzrgb")))
        if ligand_out["point_cloud_size_2sigma"] == 0:
            # print("Warning: No points for ligand", lig_name, "in 2 sigma contour")
            ligand_out["missing_ligand"] = "ErrorPc2sigmaParsing"
            # error_ligands.append(lig_name + '_sigma_2')
        ligand_out["point_cloud_size_3sigma"] = rawcount(
            (lig_output_path / (lig_name + "_lig_point_cloud_fofc_3.0sigma.xyzrgb")))
        if ligand_out["point_cloud_size_3sigma"] == 0:
            # print("Warning: No points for ligand", lig_name, "in 3 sigma contour")
            ligand_out["missing_ligand"] = "ErrorPc3sigmaParsing"
            # error_ligands.append(lig_name + '_sigma_3')
        pbar.update(1)
        return ligand_out

    # create the ligand output path if it does not exists
    if not lig_output_path.exists() or not lig_output_path.is_dir():
        lig_output_path.mkdir(parents=True)

    # print("\n* Start processing ligand entry", lig_name, "-", i + 1, "/", n, "*")
    t1 = time.time()
    # get the ligand label file path
    lig_label_pos_file = db_ligxyz_path / (lig_name + '_class.xyz')
    # get the pdb entry electron density mean and std
    median_I = lig_data["rho_median"]
    q25th_I = lig_data["rho_q25th"]
    q75th_I = lig_data["rho_q75th"]
    grid_space = lig_data["grid_space"]

    try:
        # read ligand coordinate file and labels
        lig_label_pos = pd.read_csv(lig_label_pos_file, skipinitialspace=True)
        # read grid point cloud fofc and select the values around the sigma 2 mask with 2A expansion
        lig_grid = o3d.io.read_point_cloud(
            (lig_grid_path / (lig_name + "_grid_point_cloud_fofc.xyzrgb")).as_posix())
        # print("Number of points in the ligand grid", len(lig_grid.points))
        # NOT DOING it - select only positive density values in sigma scale, e.g. value above the p mean
        # lig_grid = extract_pc_gt_value(lig_grid, 0)[0]
        # print("Number of positive points in the ligand grid", len(lig_grid.points))

        # get the local min and max intensity
        # Computing min and max rho in the entire ligand grid, use only one of the colors (they have the same values in all chanels)
        # min_I, max_I = min_max(np.asarray(lig_grid.colors)[:, 0])
        # print("Local grid rho min = ", min_I, "\nLocal grid rho max = ", max_I)

        # print("- Extract the ligand point cloud from the grid by applying a 2 sigma mask with ", expansion_radii,
        #       "A expansion")
        lig2sigmaIdx = extract_lig_pc_sigma_idx(lig_grid, median_I, q25th_I, q75th_I, 2.0, lig_label_pos,
                                                grid_space)
        lig_fofc = extract_pc_neighborhood(lig_grid, lig2sigmaIdx, expansion_radii)
        # print("Number of points in the ligand point cloud 2 sigma mask", len(lig_fofc.points))
        # draw the geometries
        if test_mode:
            print("Plot rho grid points distribution")
            plot_dist(np.asarray(lig_grid.colors)[:, 0])
            o3d.visualization.draw_geometries([lig_fofc])
        del lig_grid, lig2sigmaIdx
    except Exception as e:
        # print("Error selecting positive values above the sigma 2 mask with ", expansion_radii,
        #       "A expansion from the ligand", lig_name,
        #       "point cloud boundaries.")
        # print("Error message:", e)
        ligand_out["missing_ligand"] = "ErrorPc2sigmaMaskCreation"
        # error_ligands.append(lig_name+"_2sigmaMask")
        pbar.update(1)
        return ligand_out

    # print("- Label the ligand's points using the distance to the ligand's atoms")
    try:
        # label points according to their distance to the closest atom and the respective atom radii
        lig_kdtree = o3d.geometry.KDTreeFlann(lig_fofc)

        # store the distance to the atom used to color each point
        points_labeled_dist = np.full(len(lig_fofc.points), -1.0)
        points_label = np.full(len(lig_fofc.points), "-1")
        # if test mode save original density values
        if test_mode:
            p_sigma = np.array(lig_fofc.colors, copy=True)
            plot_dist(p_sigma[:, 0])

        for atom_i in range(lig_label_pos.shape[0]):
            # select points that are within the atom radii from the current atom position
            neighbors_points = list(
                lig_kdtree.search_radius_vector_3d(np.array(lig_label_pos[['x', 'y', 'z']].iloc[atom_i, :]),
                                                   elements_radii[lig_label_pos.symbol[atom_i]]))
            neighbors_points[2] = np.asarray(neighbors_points[2])
            # assign distance to points if they are not labeled yet or if the distance to this atom is smaller
            points_selected_label = ((points_labeled_dist[neighbors_points[1]] == -1) |
                                     (points_labeled_dist[neighbors_points[1]] > neighbors_points[2]))
            idx_selected_pts = np.asarray(neighbors_points[1])[points_selected_label]
            # update minimum distance to each reached point
            points_labeled_dist[idx_selected_pts] = neighbors_points[2][points_selected_label]
            # select atoms that are within the atom radii to be labeled with the atom label
            points_label[idx_selected_pts] = lig_label_pos.labels[atom_i]
            # only color in test mode
            if test_mode:
                np.asarray(lig_fofc.colors)[idx_selected_pts, :] = elements_color_SP_test[
                    str(lig_label_pos.labels[atom_i])]

        if test_mode:
            o3d.visualization.draw_geometries([lig_fofc])
            # restore original density values if in test mode
            np.asarray(lig_fofc.colors)[:, :] = p_sigma
    except Exception as e:
        # print("Error labeling the ligand", lig_name, "point cloud.")
        # print("Error message:", e)
        ligand_out["missing_ligand"] = "ErrorPcLabeling"
        # error_ligands.append(lig_name+'_label')
        pbar.update(1)
        return ligand_out
    if len(lig_fofc.points) != len(points_label):
        print("Wrong number of point labels. Not matching with the total number of points")
        sys.exit(1)

    # save the ligand points labels
    np.savetxt((lig_output_path / (lig_name + "_lig_pc_labels_2sigmaMask.txt")).as_posix(),
               points_label, fmt="%s")
    # print("Number of point labels", len(points_label))
    ligand_out["point_cloud"] = True
    ligand_out["point_cloud_size"] = len(lig_fofc.points)
    #
    # print("- Sigma scale the electron density values inside the 2 sigma Mask")
    # extract ligand point cloud for 2 sigma Mask, 0.5, 1, 2 and 3 sigma images
    # scale the electron density rho using sigma, then apply the contours
    p_sigma = robust_sigma_scale_clipping(lig_fofc.colors, median_I, q25th_I, q75th_I)
    # sigma scale min and max intensities
    min_I, max_I = min_max(p_sigma)
    # print("Sigma scaled local rho min = ", min_I, ", rho max = ", max_I)
    # scale from 0 to 1 using max and min rho in the  2 sigma mask pc
    np.asarray(lig_fofc.colors)[:, :] = min_max_scale(p_sigma, min_I, max_I)

    # 2 sigma mask
    # print("Create the ligand point cloud with sigma scaled electron density values")
    if test_mode:
        o3d.visualization.draw_geometries([lig_fofc])
        print("Plot rho sigma scales distribution")
        plot_dist(p_sigma[:, 0])
    o3d.io.write_point_cloud((lig_output_path / (lig_name + "_lig_point_cloud_fofc_2sigmaMask.xyzrgb")).as_posix(),
                             lig_fofc)
    # min max scale each image according to the contour minimum and maximum
    sigma_05_mask = create_lig_pc_sigma(p_sigma, lig_fofc, points_label, 0.5, lig_label_pos, grid_space,
                                        lig_output_path, lig_name)
    ligand_out["point_cloud_size_05sigma"] = sum(sigma_05_mask)
    # 1 sigma
    sigma_1_mask = create_lig_pc_sigma(p_sigma, lig_fofc, points_label, 1.0, lig_label_pos, grid_space,
                                       lig_output_path, lig_name)
    ligand_out["point_cloud_size_1sigma"] = sum(sigma_1_mask)
    # 2 sigma
    sigma_2_mask = create_lig_pc_sigma(p_sigma, lig_fofc, points_label, 2.0, lig_label_pos, grid_space,
                                       lig_output_path, lig_name)
    ligand_out["point_cloud_size_2sigma"] = sum(sigma_2_mask)
    # 3 sigma
    sigma_3_mask = create_lig_pc_sigma(p_sigma, lig_fofc, points_label, 3.0, lig_label_pos, grid_space,
                                       lig_output_path, lig_name)
    ligand_out["point_cloud_size_3sigma"] = sum(sigma_3_mask)

    # make image with different values in each chanel - all values, 1 and 2 sigma
    # min max scale each chanel using the contour minimum and the values maximum
    # print("Create ligand point cloud with all,1,2 sigma contour in each chanel")
    # red if point only appears in the mask, yellow if appears in 1 sigma, and grey scale if appears in 1 and 2 sigma contour
    if sigma_1_mask.any():
        p_sigma[:, 1][sigma_1_mask == False] = 1.0
        p_sigma[:, 1] = min_max_scale(p_sigma[:, 1], 1.0, max_I)
    else:
        # print("Warning: No points for ligand", lig_name, "in 1 sigma contour")
        ligand_out["missing_ligand"] = "ErrorPc1sigmaCreation"
        # error_ligands.append(lig_name+'_sigma_1')
        pbar.update(1)
        return ligand_out
    if sigma_2_mask.any():
        p_sigma[:, 2][sigma_2_mask == False] = 2.0
        p_sigma[:, 2] = min_max_scale(p_sigma[:, 2], 2.0, max_I)
    else:
        # print("Warning: No points for ligand", lig_name, "in 2 sigma contour")
        ligand_out["missing_ligand"] = "ErrorPc2sigmaCreation"
        # error_ligands.append(lig_name + '_sigma_2')
        pbar.update(1)
        return ligand_out

    np.asarray(lig_fofc.colors)[:, 1:] = p_sigma[:, 1:]
    if test_mode:
        o3d.visualization.draw_geometries([lig_fofc])

    o3d.io.write_point_cloud(
        (lig_output_path / (lig_name + "_lig_point_cloud_fofc_1_2sigmaMask.xyzrgb")).as_posix(),
        lig_fofc)

    # make image with different values in each chanel - all values, 2 and 3 sigma
    # print("Create ligand point cloud with all,2,3 sigma contour in each chanel")
    # red if point only appears in the mask, yellow if appears in 2 sigma, and grey scale if appears in 2 and 3 sigma contour
    p_sigma[:, 1] = p_sigma[:, 2]

    if sigma_3_mask.any():
        p_sigma[:, 2][sigma_3_mask == False] = 3.0
        p_sigma[:, 2][sigma_3_mask] = p_sigma[:, 0][sigma_3_mask]
        p_sigma[:, 2] = min_max_scale(p_sigma[:, 2], 3.0, max_I)
    else:
        # print("Warning: No points for ligand", lig_name, "in 3 sigma contour")
        ligand_out["missing_ligand"] = "ErrorPc3sigmaCreation"
        # error_ligands.append(lig_name + '_sigma_3')
        pbar.update(1)
        return ligand_out

    np.asarray(lig_fofc.colors)[:, 1:] = p_sigma[:, 1:]
    if test_mode:
        o3d.visualization.draw_geometries([lig_fofc])

    o3d.io.write_point_cloud(
        (lig_output_path / (lig_name + "_lig_point_cloud_fofc_2_3sigmaMask.xyzrgb")).as_posix(),
        lig_fofc)

    d = time.time() - t1
    # print("Processed ligand", lig_name, "in: %.2f s." % d)
    ligand_out['time_process'] = d
    pbar.update(1)
    return ligand_out


class EngineLigandPointCloudSigmaScale(object):
    def __init__(self, ligs_grid_path, output_path, db_ligxyz_path, grid_space, overwrite_pc, pbar):
        self.ligs_grid_path = ligs_grid_path
        self.output_path = output_path
        self.db_ligxyz_path = db_ligxyz_path
        self.grid_space = grid_space
        self.overwrite_pc = overwrite_pc
        self.pbar = pbar
    def __call__(self, lig_data_row):
        lig_out = create_ligand_point_cloud_sigma_scaled(lig_data_row, self.ligs_grid_path, self.output_path,
                                                         self.db_ligxyz_path, self.grid_space, self.overwrite_pc,
                                                         self.pbar)
        return lig_out


def create_ligands_dataset_point_cloud_sigma_scaled(db_ligxyz_path, ligs_grid_path, output_path, num_processors = 2,
                                                    grid_space=0.2, overwrite_pc=False, i_start=0, n=0):
    if n <= i_start and n > 0:
        sys.exit("Wrong range: The provided stop row is not greater than the start row.")
    t0 = time.time()
    # check inputs
    db_ligxyz_path = Path(db_ligxyz_path)
    valid_ligs_file = db_ligxyz_path / (db_ligxyz_path.name.replace("xyz_", "") + '_box_pc.csv')
    # grid spacing
    grid_space = float(grid_space)

    if not db_ligxyz_path.exists() or not db_ligxyz_path.is_dir():
        sys.exit("The provided ligand xyz data folder do not exists.")
    if not Path(valid_ligs_file).is_file():
        sys.exit("The valid ligands list CSV file do not exists in the provided ligand xyz data folder.")

    ligs_grid_path = Path(ligs_grid_path)
    if not ligs_grid_path.exists() or not ligs_grid_path.is_dir():
        sys.exit("The provided ligand point cloud grids data folder do not exists.")

    output_path = Path(output_path)
    if not output_path.exists() or not output_path.is_dir():
        output_path.mkdir(parents=True)

    # read csv with the valid ligands list to encode
    ligs_retrieve = pd.read_csv(valid_ligs_file, na_values = ['null', 'N/A'], keep_default_na = False,
                                usecols = ['ligID', 'ligCode', 'entry', 'filter_quality', 'x', 'y', 'z',
                                           'x_bound', 'y_bound', 'z_bound', 'grid_space',
                                           'rho_median', 'rho_q25th', 'rho_q75th']) # do not interpret sodium NA as nan
    # filter out the low quality ligands
    ligs_retrieve = ligs_retrieve[ligs_retrieve.filter_quality]
    if n == 0 or n > ligs_retrieve.shape[0]:
        n = ligs_retrieve.shape[0]

    # run the ligand pc creation in multiprocessing
    pbar = tqdm(total=ligs_retrieve.shape[0])
    try:
        pool_ligands = ThreadPool(num_processors)
        # create the engine to create the ligand pc passing the parameters
        engine_createLigsPc = EngineLigandPointCloudSigmaScale(ligs_grid_path, output_path, db_ligxyz_path,
                                                               grid_space, overwrite_pc, pbar)
        ligands_out = pool_ligands.map(engine_createLigsPc, ligs_retrieve.iloc[i_start:n, :].iterrows())
    finally: # To make sure processes are closed in the end, even if errors happen
        pool_ligands.close()
        pool_ligands.join()
    # convert the result in a dataframe and merge with the ligands data
    ligands_out = pd.DataFrame(ligands_out)
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

    error_ligands = ligs_retrieve.missing_ligand.str.match("ErrorPcLabeling|.*sigmaCreation|.*sigmaParsing|ErrorPc2sigmaMaskCreation")
    if sum(error_ligands) > 0:
        print("- Error ligands in labeling, point cloud in sigma creation and/or parsing:")
        print(ligs_retrieve.ligID[error_ligands].values)

    if sum(entries_missing) > 0:
        print("A total of "+str(len(ligs_retrieve.entry[entries_missing].unique()))+"/",
              str(len(ligs_retrieve[i_start:n].entry.unique())),
              " PDB entries point clouds data was missing.")

    if sum(skipped_ligands) > 0:
        print("A total of "+str(sum(skipped_ligands))+"/", str(n-i_start),
              " ligands entries were skipped due to missing PDB entries point clouds data.")

    if sum(error_ligands) > 0:
        print("A total of "+str((ligs_retrieve["point_cloud"] != True).sum())+"/", str(n-i_start),
              " ligands entries were missing or raised an error and could not be entirely processed.")

    # remove failed ligands, save csv of the created point clouds and print number of pdb and ligand errors
    ligs_retrieve = ligs_retrieve[ligs_retrieve["point_cloud"]]
    ligs_retrieve.to_csv((db_ligxyz_path / valid_ligs_file.name.replace(".csv", "_sigma_scale.csv")).as_posix())
    d = time.time() - t0

    print("\nDONE!", "Time elapsed: %.2f s." % d)


if __name__ == "__main__":
    # read the ligands csv with respective pdb entries, x, y, z and data
    overwrite_pc = False
    i_start = n = 0
    grid_space = 0.2
    if len(sys.argv) >= 5:
        db_ligxyz_path = sys.argv[1]
        ligs_grid_path = sys.argv[2]
        output_path = sys.argv[3]
        num_processors = int(sys.argv[4])
        if len(sys.argv) >= 6:
            grid_space = float(sys.argv[5])
        if len(sys.argv) >= 7:
            overwrite_pc = (sys.argv[6].lower() == "true")
        if len(sys.argv) >= 8:
            i_start = int(sys.argv[7])
        if len(sys.argv) >= 9:
            n = int(sys.argv[8])

    else:
        sys.exit("Wrong number of arguments. Three argument must be supplied in order to read the ligands "
                 "mtz file, xyz file and output path to store the extracted point cloud in sigma scale: \n"
                 "  1. The path to the data folder called 'xyz_<ligand csv name>' where the ligands .xyz files with "
                 "their labels are located. It must also contain the CSV file with the valid ligands list for which a grid point cloud "
                 "was created. This file is named as '<ligand csv name>_box_class_freq_pc.csv' and is expected to be the output "
                 "of the mtz to grid point cloud script. "
                 "Mandatory columns = 'ligID', 'ligCode', 'entry', 'rho_median', 'rho_q25th', 'rho_q75th'.;\n"
                 "  2. The path to the folder where the point clouds grids are stored ('data/point_clouds').\n"
                 "  3. The path to the output folder where the point clouds in sigma scale will be stored ('data/point_clouds' or other).\n"
                 "  4. The number of processors to use for multiprocessing parallelization;\n"
                 "  5. (optional) The grid space size in angstroms to be used to create the point clouds. "
                 "Default to 0.2 A;\n"
                 "  6. (optional) A boolean True or False indicating if the already processed ligands should be "
                 "overwritten (Default to False).\n"
                 "  7. (optional) The number of the row of the ligands CSV file where the script should start. "
                 "Skip to the given row or, if missing, start from the beginning;\n"
                 "  8. (optional) The number of the row of the ligands CSV file where the script should stop. "
                 "Stop in the given row or, if missing, equal zero or greater than the number of rows, stop in the last row.\n"
                 )
    create_ligands_dataset_point_cloud_sigma_scaled(db_ligxyz_path, ligs_grid_path, output_path, num_processors,
                                                    grid_space, overwrite_pc, i_start, n)
