from pathlib import Path
import numpy as np
import open3d as o3d
import sys
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

# the Van der Waals Radii of Elements from  S. S. Batsanov 2001
# elements_radii = {'B': 1.8, 'C': 1.7,
#                   'N': 1.6, 'P': 1.95,
#                   'O': 1.55, 'S': 1.8,
#                   'Cl': 1.8, 'F': 1.5, 'Br': 1.9, 'I': 2.1}
# the experimental Van der Waals Radii averaged from 1.5A to 1.8A rounded in 2 decimals from XGen 2020
# - B received S and Se received Br radii
elements_radii = {'B': 1.4, 'C': 1.46,
                  'N': 1.44, 'P': 1.4,
                  'O': 1.42, 'S': 1.4, 'Se': 1.37,
                  'Cl': 1.4, 'F': 1.4, 'Br': 1.37, 'I': 1.37}
# for testing with visual inspection color the points according to their label
elements_color_SP_test = {'0': np.array([237, 238, 192])/255, '1': np.array([67, 62, 14])/255,
                          '2': np.array([124, 144, 130])/255, '3': np.array([167, 162, 132])/255,
                          '4': np.array([208, 200, 142]) / 255, '5': np.array([242, 132, 130])/255,
                          '6': np.array([255, 136, 17]) / 255, '7': np.array([57, 47, 90])/255,
                          '8': np.array([84, 87, 124]) / 255, '9': np.array([220, 127, 155])/255,
                          '10': np.array([150, 173, 200]) / 255, '11': np.array([167, 194, 193])/255,
                          '12': np.array([183, 214, 186]) / 255, '13': np.array([215, 255, 171])/255,
                          '14': np.array([234, 255, 140]) / 255, '15': np.array([252, 255, 108])/255,
                          '16': np.array([216, 157, 106]) / 255, '17': np.array([163, 113, 91])/255,
                          '18': np.array([109, 69, 76]) / 255, '19': np.array([122, 86, 92])/255,
                          '20': np.array([59, 31, 43]) / 255, '21': np.array([219, 22, 47]) / 255,
                          '22': np.array([0, 0, 1]), '23': np.array([0,1,0]),
                          '24': np.array([1,0,0]), '25': np.array([1,0,1]),
                          '26': np.array([1,1,0]), '-1': np.array([0,0,0])}

def plot_dist(data, chanel=0):
    if chanel > 0:
        log_scale = [False,True]
        data = data+0.0001
    else:
        log_scale = False
    sns.set_theme(style="darkgrid")
    g = sns.displot(data, height=5, log_scale=log_scale)
    # plt.show()
    plt.savefig('distribution_rho'+str(chanel)+'.png')

def check_ligands_point_clouds(db_ligxyz_path, pc_data_path, draw_pc=True, grid_space=0.2):
    # check inputs and create missing directories
    pc_data_path = Path(pc_data_path)
    db_ligxyz_path = Path(db_ligxyz_path)
    valid_ligs_file = db_ligxyz_path / (db_ligxyz_path.name.replace("xyz_", "") + '_box_class_freq.csv')

    if not db_ligxyz_path.exists() or not db_ligxyz_path.is_dir():
        sys.exit("The provided ligand xyz data folder do not exists.")
    if not pc_data_path.exists() or not pc_data_path.is_dir():
        sys.exit("The provided point cloud data folder do not exists.")
    if not Path(valid_ligs_file).is_file():
        sys.exit("The provided valid ligands list CSV file do not exists.")

    # set grid space
    grid_space = float(grid_space)
    elements_sphere_points = {e: 4 / 3 * np.pi * np.power(elements_radii[e], 3) / np.power(grid_space, 3) for e in
                              elements_radii}
    # read csv with the valid ligands list to encode
    ligs_retrieve = pd.read_csv(valid_ligs_file, na_values = ['null', 'N/A'], keep_default_na = False) # do not interpret sodium NA as nan
    # filter out the low quality ligands
    ligs_retrieve = ligs_retrieve[ligs_retrieve.filter_quality]
    n = ligs_retrieve.shape[0]
    # sort by ligand Code, to test the same ligands together
    ligs_retrieve = ligs_retrieve.sort_values(['ligCode', 'missingHeavyAtoms']).reset_index(drop=True)
    # store the errors and point clouds sizes
    error_ligands = []
    missing_ligands = []
    ligs_retrieve["error"] = False
    ligs_retrieve["missing_data"] = False
    ligs_retrieve["point_cloud"] = False
    ligs_retrieve["point_cloud_size_2sigmaMask"] = -1
    ligs_retrieve["point_cloud_size_0.5sigma"] = -1
    ligs_retrieve["point_cloud_size_1.0sigma"] = -1
    ligs_retrieve["point_cloud_size_2.0sigma"] = -1
    ligs_retrieve["point_cloud_size_3.0sigma"] = -1
    ligs_retrieve["point_cloud_pNoLabelPoints_2sigmaMask"] = -1
    ligs_retrieve["point_cloud_pNoLabelPoints_0.5sigma"] = -1
    ligs_retrieve["point_cloud_pNoLabelPoints_1.0sigma"] = -1
    ligs_retrieve["point_cloud_pNoLabelPoints_2.0sigma"] = -1
    ligs_retrieve["point_cloud_pNoLabelPoints_3.0sigma"] = -1
    # create the grid and point cloud for each valid ligand
    for i in range(n):
        entry_name = ligs_retrieve.loc[i, "entry"]
        # get the ligand id and its label file path
        lig_name = ligs_retrieve.loc[i, 'ligID']
        lig_label_pos_file = db_ligxyz_path / (lig_name + '_class.xyz')
        print("\n* Start processing ligand entry", lig_name, "-", i, "/", n, "*")
        t0 = time.time()
        # for each ligand check if the point clouds exists, draw them, check the consistency of the labels and
        # draw the pc colored by the labels
        #
        # validate lig pc path
        lig_pc_path = pc_data_path / entry_name
        if not lig_pc_path.exists():
            missing_ligands.append(lig_name + '_PDB_pc_missing')
            ligs_retrieve.loc[i, "missing_data"] = True
            print("Error: PDB entry ", entry_name, "point cloud folder is missing.")
            continue
        # elif not (lig_pc_path / (lig_name+"_grid_point_cloud_fofc.xyzrgb")).exists() and \
        #      not (lig_pc_path / (lig_name+"_grid_point_cloud_fofc.xyzrgb")).is_file():
        #     missing_ligands.append(lig_name + '_grid_pc_missing')
        #     ligs_retrieve.loc[i, "missing_data"] = True
        #     print("Error: Ligand ", lig_name, "grid point cloud file is missing.")
        #     continue

        # draw the grid
        # print("  - Ligand grid point cloud")
        # lig_fofc = o3d.io.read_point_cloud((lig_pc_path / (lig_name+"_grid_point_cloud_fofc.xyzrgb")).as_posix())
        # o3d.visualization.draw_geometries([lig_fofc])
        # check if the point clouds exists (0,05,1,2,0_1_2 sigma, draw them
        # # check the labels and draw the pc again colored with the labels
        #
        # read ligand coordinate file and labels
        lig_label_pos = pd.read_csv(lig_label_pos_file, skipinitialspace=True)
        # check labels quantity
        lig_label_pos.labels = lig_label_pos.labels.astype(str)
        if not lig_label_pos.labels.value_counts().to_dict() == ligs_retrieve.loc[i, "0":][
            ligs_retrieve.loc[i, "0":] > 0].to_dict():
            error_ligands.append(lig_name + "_pcLabels_wrongCount")
            ligs_retrieve.loc[i, "error"] = True
            print("Error: Ligand ", lig_name, " points labels count do not match.")
        # for each sigma contour pc of the ligand: 0sigma, 0.5sigma, 1.0sigma, 2.0sigma, 0_1_2sigma, 2_3sigmaMask
        for sigma_contour in ["1_2sigmaMask", "2.0sigma", "1.0sigma", "0.5sigma", "3.0sigma",  "2_3sigmaMask"]:
            # sigma_contour = "0sigma"
            print("  - Ligand ", sigma_contour," point cloud")
            pc_lig = lig_pc_path / (lig_name + "_lig_point_cloud_fofc_"+sigma_contour+".xyzrgb")
            if "Mask" not in sigma_contour:
                p_labels_lig = lig_pc_path / (lig_name + "_lig_pc_labels_"+sigma_contour+".txt")
            else:
                p_labels_lig = lig_pc_path / (lig_name + "_lig_pc_labels_2sigmaMask" + ".txt")

            if not pc_lig.exists() and not pc_lig.is_file():
                missing_ligands.append(lig_name + "_"+sigma_contour+"_pc_missing")
                ligs_retrieve.loc[i, "missing_data"] = True
                print("Error: Ligand ", lig_name, sigma_contour," point cloud file is missing.")
                continue
            if not p_labels_lig.exists() and not p_labels_lig.is_file():
                missing_ligands.append(lig_name + "_"+sigma_contour+"_pcLabels_missing")
                ligs_retrieve.loc[i, "missing_data"] = True
                print("Error: Ligand ", lig_name, sigma_contour," points labels file is missing.")
                continue

            lig_fofc = o3d.io.read_point_cloud(pc_lig.as_posix())
            if "Mask" not in sigma_contour:
                ligs_retrieve.loc[i, "point_cloud_size_"+sigma_contour] = len(lig_fofc.points)
                print("    - Size:", ligs_retrieve.loc[i, "point_cloud_size_"+sigma_contour], "points")
            else:
                ligs_retrieve.loc[i, "point_cloud_size_2sigmaMask"] = len(lig_fofc.points)
                print("    - Size:", ligs_retrieve.loc[i, "point_cloud_size_2sigmaMask"], "points")

            if len(lig_fofc.points) == 0:
                missing_ligands.append(lig_name + "_" + sigma_contour + "_pc_missing")
                ligs_retrieve.loc[i, "missing_data"] = True
                print("Error: Ligand ", lig_name, sigma_contour, " point cloud file is empty.")
                continue

            if draw_pc:
                plot_dist(np.asarray(lig_fofc.colors)[:,0],0)
                if "Mask" in sigma_contour:
                    plot_dist(np.asarray(lig_fofc.colors)[:, 1], 1)
                    plot_dist(np.asarray(lig_fofc.colors)[:, 2], 2)
                o3d.visualization.draw_geometries([lig_fofc])

            print("Descriptive statistics of the pc features")
            print(pd.DataFrame(np.asarray(lig_fofc.colors)).describe())
            print()

            print("    - Check ligand's points labels and the number of points around each atom")
            lig_labels = np.asarray([line.rstrip('\n') for line in open(p_labels_lig)])
            try:
                # label points according to their distance to the closest atom and the respective atom radii
                lig_kdtree = o3d.geometry.KDTreeFlann(lig_fofc)
                atoms_points = []
                atoms_error = False
                for atom_i in range(lig_label_pos.shape[0]):
                    # select points that are within each element radii from the current atom
                    neighbors_points = list(lig_kdtree.search_radius_vector_3d(np.array(lig_label_pos[['x','y','z']].iloc[atom_i,:]),
                                                                               elements_radii[lig_label_pos.symbol[atom_i]]))
                    idx_selected_pts = np.asarray(neighbors_points[1])
                    atoms_points.append((len(idx_selected_pts)-1)/elements_sphere_points[lig_label_pos.symbol[atom_i]])
                    # print("      #"+str(atom_i), "atom ", lig_label_pos.symbol[atom_i], ": have",
                    #       str(len(idx_selected_pts)/elements_sphere_points[lig_label_pos.symbol[atom_i]]*100),
                    #       "% of the expected number of points within its radii distance")
                    neighbors_points[2] = np.asarray(neighbors_points[2])
                    # select the closer points to the atom center (< half of its radii) to check the labels consistency
                    points_selected_label = idx_selected_pts[
                        neighbors_points[2] < np.power(elements_radii[lig_label_pos.symbol[atom_i]]/4, 2)]
                    # check this points labels consistency
                    if not (lig_labels[points_selected_label] == str(lig_label_pos.labels[atom_i])).all():
                        atoms_error = True
                        print("Error: Ligand ", lig_name, sigma_contour, " atom", atom_i, "have wrong labeled points")

                if atoms_error:
                    error_ligands.append(lig_name + "_" + sigma_contour + "_wrong_atoms_label")
                    ligs_retrieve.loc[i, "error"] = True

                print("      - The atoms have on average ", np.mean(atoms_points)*100,
                      "% of the expected number of points within their radii distance")
                # color using labels
                if draw_pc:
                    np.asarray(lig_fofc.colors)[:, :] = [elements_color_SP_test[x] for x in lig_labels]
                    o3d.visualization.draw_geometries([lig_fofc])
                # store mean number of covered atom points and percentage of No label points in this image
                if "Mask" not in sigma_contour:
                    ligs_retrieve.loc[i, "atoms_points_cover_" + sigma_contour] = np.mean(atoms_points)
                    ligs_retrieve.loc[i, "point_cloud_pNoLabelPoints_" + sigma_contour] = (lig_labels == "-1").sum()/len(lig_labels)
                else:
                    ligs_retrieve.loc[i, "atoms_points_cover_2sigmaMask"] = np.mean(atoms_points)
                    ligs_retrieve.loc[i, "point_cloud_pNoLabelPoints_2sigmaMask"] = (lig_labels == "-1").sum() / len(lig_labels)
            except Exception as e:
                print("Error checking the ligand point cloud labels. Ligand ID:",
                      lig_name, e)
                error_ligands.append(lig_name + "_"+sigma_contour+"_pcLabels_error")
                ligs_retrieve.loc[i, "error"] = True
                continue

        ligs_retrieve.loc[i, "point_cloud"] = True
        d = time.time() - t0
        print("Checked ligand",lig_name, "in: %.2f s." % d)

   # save csv of the checked point clouds and print number of pdb and ligand errors
    ligs_retrieve.to_csv(valid_ligs_file.as_posix().replace('.csv', '_tested_pc.csv'))

    if len(missing_ligands) > 0:
        print("- Missing ligands files:")
        print(str(missing_ligands))

    if len(error_ligands) > 0:
        print("- Error ligands:")
        print(str(error_ligands))

    if len(missing_ligands) > 0:
        print("A total of " + str(ligs_retrieve.missing_data.sum()) + "/", str(n),
              " ligands entries had at least one missing data.")

    if len(error_ligands) > 0:
        print("A total of "+str(ligs_retrieve.error.sum())+"/", str(n),
              " ligands entries raised an error in the checking procedure.")

    print("DONE!")


if __name__ == "__main__":
    grid_space = 0.2
    draw_pc = True
    if len(sys.argv) >= 3:
        db_ligxyz_path = sys.argv[1]
        pc_data_path = sys.argv[2]
        if len(sys.argv) >= 4:
            draw_pc = (sys.argv[3].lower() == "true")
        if len(sys.argv) >= 5:
            grid_space = float(sys.argv[4])
    else:
        sys.exit("Wrong number of arguments. Two argument must be supplied in order to check a list of ligands "
                 "point clouds size and labels: \n"
                 "  1. The path to the data folder called 'xyz_<ligand csv name>' where the ligands .xyz files with "
                 "their labels are located. It must also contain the CSV file with the valid ligands list and their bounding box "
                 "position. This file is named as '<ligand csv name>_box_class_freq.csv' and is expected to be the output "
                 "of the endoce ligands script. "
                 "Mandatory columns = 'ligID', 'ligCode', 'entry', 'filter_quality', 'x', 'y', 'z', 'x_bound', 'y_bound',"
                 "'z_bound'.;\n"
                 "  2. The path to the data folder where the ligands point clouds are located ('data/point_clouds').\n"
                 "  3. (Optional) Boolean True or False to draw point clouds. Default to True\n"
                 "  4. (Optional) The grid space used to create the point clouds. Default to 0.2."
                 )
    check_ligands_point_clouds(db_ligxyz_path, pc_data_path, draw_pc, grid_space)
