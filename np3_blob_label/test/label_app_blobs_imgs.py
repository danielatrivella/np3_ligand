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
        # expansion_radii = expansion_radii*1.1
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
