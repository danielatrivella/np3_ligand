from pathlib import Path
import numpy as np
import open3d as o3d
import sys
import pandas as pd

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
    # convert tabel to dict
    dict_reso_atomic_radii = radii_table.to_dict()
    # set atomic radii outside the range defined in the xgen table,
    # # set them as the first limit to the right and to the left
    for radii in np.arange(0.5, 1.0, 0.1):
        dict_reso_atomic_radii[str(np.round(radii, 1))] = dict_reso_atomic_radii["1.0"]
    for radii in np.arange(3.1, 4.1, 0.1):
        dict_reso_atomic_radii[str(np.round(radii, 1))] = dict_reso_atomic_radii["3.0"]
    # return the table of atomic radii by resolution as a dictionary
    return dict_reso_atomic_radii
    
np3_atomic_radii_table_path = Path(Path(__file__).parent, "..", "..", "np3_LigPCDS", "atomic_radii_tables.csv")
elements_radii_w_reso = read_radii_table_to_dict(np3_atomic_radii_table_path)

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

def label_blob_imgs(blob_img_path, lig_label_path, blob_id, reso):
    # round resolution to 1 decimal place
    reso = str(round(float(reso), 1))
    # read the blob img file
    blob_fofc = o3d.io.read_point_cloud(blob_img_path.as_posix())
    # read the ligand label file
    lig_label_pos = pd.read_csv(lig_label_path, skipinitialspace=True)
    # print("- Label the ligand's points using the distance to the ligand's atoms")
    try:
        # label points according to their distance to the closest atom and the respective atom radii
        blob_kdtree = o3d.geometry.KDTreeFlann(blob_fofc)
        # store the distance to the atom used to color each point
        points_labeled_dist = np.full(len(blob_fofc.points), -1.0)
        points_label = np.full(len(blob_fofc.points), "-1")
        # expand mask in 10% to deal with atoms contours that are greater than the atom radii
        # if test mode save original density values
        if test_mode:
            p_q_rank = np.array(blob_fofc.colors, copy=True)
            # plot_dist(p_q_rank[:, 0])
        for atom_i in range(lig_label_pos.shape[0]):
            # select points that are within the atom radii from the current atom position+expansion radii
            neighbors_points = list(
                blob_kdtree.search_radius_vector_3d(np.array(lig_label_pos[['x', 'y', 'z']].iloc[atom_i, :]),
                                                   elements_radii_w_reso[reso][lig_label_pos.symbol[atom_i]] +
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
                np.asarray(blob_fofc.colors)[idx_selected_pts, :] = elements_color_SP_test[
                    str(lig_label_pos.labels[atom_i])]

        if test_mode:
            print("draw colored labels")
            o3d.visualization.draw_geometries([blob_fofc])
            # restore original density values if in test mode
            np.asarray(blob_fofc.colors)[:, :] = p_q_rank[points_labeled_dist > -1]
    except Exception as e:
        print("Error labeling the blob", blob_id, "point cloud.")
        print("Error message:", e)
        # error_ligands.append(lig_name+'_label')
        return False
    if len(blob_fofc.points) != len(points_label):
        print("Wrong number of point labels. Not matching with the total number of points")
        sys.exit(1)

    # save label file
    np.savetxt((blob_img_path.parent / (blob_id + "_lig_pc_labels_qRankMask.txt")).as_posix(),
               points_label, fmt="%s")
    return True
