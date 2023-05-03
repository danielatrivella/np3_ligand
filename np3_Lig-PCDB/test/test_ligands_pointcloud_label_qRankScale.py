from pathlib import Path
import numpy as np
import open3d as o3d
import sys
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from p_tqdm import p_map
from multiprocessing import cpu_count

def read_radii_table_to_dict(radii_table_path="atomic_radii_tables.csv", radii_weight=0.65):
    # read table with atomic radii, use first row as index = atoms symbols
    radii_table = pd.read_csv(radii_table_path, index_col=0)
    # change columns names to match the resolution of each set of radii
    radii_table.columns = radii_table.columns.str.split("_").str[-1]
    # apply the weight value to the atomic radius in all resolutions
    radii_table=radii_table*radii_weight
    # return the table of atomic radii by resolution as a dictionary
    return radii_table.to_dict()

raddi_weight=0.65
# read atomic table and apply a weight of 65%
elements_radii_w_reso = read_radii_table_to_dict(radii_table_path="atomic_radii_tables.csv", radii_weight=raddi_weight)
expansion_radii = np.round(max(list(elements_radii_w_reso['2.2'].values())), 1)

# for testing with visual inspection color the points according to their labels
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


class EngineTestLigandPointCloudQRankScale(object):
    def __init__(self, db_ligxyz_path, pc_data_path, elements_sphere_points, num_processors, draw_pc):
        self.pc_data_path = pc_data_path
        self.elements_sphere_points = elements_sphere_points
        self.db_ligxyz_path = db_ligxyz_path
        self.num_processors = num_processors
        self.draw_pc = draw_pc
    def call(self, lig_data_row):
        return check_ligand_point_cloud(lig_data_row, self.db_ligxyz_path, self.pc_data_path,
                                        self.elements_sphere_points, (self.num_processors == 1), self.draw_pc)
    def run(self, ligs_retrieve):
        # pool_ligands = p_map(num_cpus=num_processors)
        ligands_out = p_map(self.call, list(ligs_retrieve.iterrows()), num_cpus=self.num_processors)
        # convert the result in a dataframe and merge with the ligands data
        ligands_out = pd.DataFrame(ligands_out)
        return ligands_out
    

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


def check_ligand_point_cloud(lig_data_row, db_ligxyz_path, pc_data_path, elements_sphere_points, 
                             verbose=False, draw_pc= False):
    lig_data_row = lig_data_row[1]
    # store the errors and point clouds sizes
    lig_name = lig_data_row['ligID']
    ligand_out = pd.Series({'ligID': lig_name})
    ligand_out["error"] = "False"
    ligand_out["missing_data"] = False
    ligand_out["point_cloud"] = False
    ligand_out["point_cloud_size_qRankMask"] = -1
    ligand_out["point_cloud_size_qRank0.5"] = -1
    ligand_out["point_cloud_size_qRank0.7"] = -1
    ligand_out["point_cloud_size_qRank0.75"] = -1
    ligand_out["point_cloud_size_qRank0.8"] = -1
    ligand_out["point_cloud_size_qRank0.85"] = -1
    ligand_out["point_cloud_size_qRank0.9"] = -1
    ligand_out["point_cloud_size_qRank0.95"] = -1
    ligand_out["point_cloud_pBackground_qRankMask"] = -1
    ligand_out["point_cloud_pBackground_qRank0.5"] = -1
    ligand_out["point_cloud_pBackground_qRank0.7"] = -1
    ligand_out["point_cloud_pBackground_qRank0.75"] = -1
    ligand_out["point_cloud_pBackground_qRank0.8"] = -1
    ligand_out["point_cloud_pBackground_qRank0.85"] = -1
    ligand_out["point_cloud_pBackground_qRank0.9"] = -1
    ligand_out["point_cloud_pBackground_qRank0.95"] = -1
    #
    # round the ligand pdb entry resolution
    reso = str(round(float(lig_data_row['Resolution']), 1))
    # create the grid and point cloud of the ligand
    entry_name = lig_data_row["entry"]
    # get the ligand label file path    
    lig_label_pos_file = db_ligxyz_path / (lig_name + '_class.xyz')
    if verbose:
        print("\n* Start processing ligand entry", lig_name, " *")
    t0 = time.time()
    # for each ligand check if the point clouds exists, draw them, check the consistency of the labels and
    # draw the pc colored by the labels
    #
    # validate lig pc path
    lig_pc_path = pc_data_path / entry_name
    if not lig_pc_path.exists():
        ligand_out["error"] = lig_name + '_PDBEntry_pc_missing'
        ligand_out["missing_data"] = True
        if verbose:
            print("Error: PDB entry ", entry_name, "point cloud folder is missing.")
        return ligand_out
    
    if verbose and (lig_pc_path / (lig_name+"_grid_point_cloud_fofc.xyzrgb")).exists() and \
            (lig_pc_path / (lig_name+"_grid_point_cloud_fofc.xyzrgb")).is_file():
        # draw the grid
        print("  - Ligand grid point cloud")
        lig_fofc = o3d.io.read_point_cloud((lig_pc_path / (lig_name + "_grid_point_cloud_fofc.xyzrgb")).as_posix())
        o3d.visualization.draw_geometries([lig_fofc])
    
    # check if the point clouds exists (0.7, 0.8, 0.8 quantile rank, draw them
    # # check the labels and draw the pc again colored with the labels
    #
    # read ligand coordinate file and labels
    lig_label_pos = pd.read_csv(lig_label_pos_file, skipinitialspace=True)
    # check labels quantity
    lig_label_pos.labels = lig_label_pos.labels.astype(str)
    if not lig_label_pos.labels.value_counts().to_dict() == lig_data_row["0":][
        lig_data_row["0":] > 0].to_dict():
        ligand_out["error"] = lig_name + "_pcLabels_wrongCount"
        if verbose:
            print("Error: Ligand ", lig_name, " point cloud's labels do not match the count.")
    # for each qRank contour pc of the ligand:
    for qRank_contour in ["qRankMask", "qRank0.5", "qRank0.7", "qRank0.75", "qRank0.8", "qRank0.85", "qRank0.9", "qRank0.95"]:
        if verbose:
            print("  - Ligand ", qRank_contour," point cloud")
        pc_lig = lig_pc_path / (lig_name + "_lig_point_cloud_fofc_"+qRank_contour+".xyzrgb")
        p_labels_lig = lig_pc_path / (lig_name + "_lig_pc_labels_"+qRank_contour+".txt")

        if not pc_lig.exists() and not pc_lig.is_file():
            ligand_out["error"] = lig_name + "_"+qRank_contour+"_pc_missing"
            ligand_out["missing_data"] = True
            if verbose:
                print("Error: Ligand ", lig_name, qRank_contour," point cloud file is missing.")
            return ligand_out
        if not p_labels_lig.exists() and not p_labels_lig.is_file():
            ligand_out["error"] = lig_name + "_"+qRank_contour+"_pcLabels_missing"
            ligand_out["missing_data"] = True
            if verbose:
                print("Error: Ligand ", lig_name, qRank_contour," points labels file is missing.")
            return ligand_out

        lig_fofc = o3d.io.read_point_cloud(pc_lig.as_posix())
        ligand_out["point_cloud_size_"+qRank_contour] = len(lig_fofc.points)
        if verbose:
            print("    - Size:", ligand_out["point_cloud_size_"+qRank_contour], "points")

        if len(lig_fofc.points) == 0:
            ligand_out["error"] = lig_name + "_" + qRank_contour + "_pc_missing"
            ligand_out["missing_data"] = True
            if verbose:
                print("Error: Ligand ", lig_name, qRank_contour, " point cloud file is empty.")
            return ligand_out

        if verbose:
            if draw_pc:
                plot_dist(np.asarray(lig_fofc.colors)[:,0],0)
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
                                                                           elements_radii_w_reso[reso][lig_label_pos.symbol[atom_i]]))
                idx_selected_pts = np.asarray(neighbors_points[1])
                atoms_points.append((len(idx_selected_pts)-1)/elements_sphere_points[reso][lig_label_pos.symbol[atom_i]])
                # print("      #"+str(atom_i), "atom ", lig_label_pos.symbol[atom_i], ": have",
                #       str(len(idx_selected_pts)/elements_sphere_points[lig_label_pos.symbol[atom_i]]*100),
                #       "% of the expected number of points within its radii distance")
                neighbors_points[2] = np.asarray(neighbors_points[2])
                # select the closer points to the atom center (< 1/4 of its radii) to check the labels consistency
                points_selected_label = idx_selected_pts[
                    neighbors_points[2] < np.power(elements_radii_w_reso[reso][lig_label_pos.symbol[atom_i]]/4, 2)]
                # check this points labels consistency
                if not (lig_labels[points_selected_label] == str(lig_label_pos.labels[atom_i])).all():
                    atoms_error = True
                    if verbose:
                        print("Error: Ligand ", lig_name, qRank_contour, " atom", atom_i, "have wrong labeled points")

            if atoms_error:
                ligand_out["error"] = lig_name + "_" + qRank_contour + "_wrong_atoms_label"
                return ligand_out

            if verbose:
                print("      - The atoms have on average ", np.mean(atoms_points)*100,
                      "% of the expected number of points within their radii distance")
                # color using labels
                if draw_pc:
                    np.asarray(lig_fofc.colors)[:, :] = [elements_color_SP_test[x] for x in lig_labels]
                    o3d.visualization.draw_geometries([lig_fofc])
            # store mean number of covered atom points and percentage of No label points in this image
            ligand_out["atoms_points_cover_" + qRank_contour] = np.mean(atoms_points)
            ligand_out["point_cloud_pBackground_" + qRank_contour] = (lig_labels == "-1").sum()/len(lig_labels)
        except Exception as e:
            if verbose:
                print("Error checking the ligand point cloud labels. Ligand ID:",
                      lig_name, e)
            ligand_out["error"] = lig_name + "_"+qRank_contour+"_pcLabels_error"
            return ligand_out

    ligand_out["point_cloud"] = True
    d = time.time() - t0
    if verbose:
        print("Checked ligand",lig_name, "in: %.2f s." % d)
    ligand_out['time_process'] = d
    return ligand_out
    

def check_ligands_point_clouds(db_ligxyz_path, pc_data_path, num_processors=2, draw_pc=True):
    t0 = time.time()
    if num_processors > cpu_count():
        num_processors = max(cpu_count() - 1, 1)
        print("Warning: The selected number of processors is greater than the available CPUs, setting it to the number of CPUs - 1 = ",
              num_processors)
    # check inputs and create missing directories
    pc_data_path = Path(pc_data_path)
    db_ligxyz_path = Path(db_ligxyz_path)
    valid_ligs_file = db_ligxyz_path / (db_ligxyz_path.name.replace("xyz_", "") + '_box_class_freq.csv')
    # read the output of the grid_to_qrankScale script to retrieve the grid spacing
    valid_ligs_pc_file = db_ligxyz_path / (db_ligxyz_path.name.replace("xyz_", "") + '_box_class_freq_qRank_scale.csv')

    if not db_ligxyz_path.exists() or not db_ligxyz_path.is_dir():
        sys.exit("The provided ligand xyz data folder do not exists.")
    if not pc_data_path.exists() or not pc_data_path.is_dir():
        sys.exit("The provided point cloud data folder do not exists.")
    if not Path(valid_ligs_file).is_file():
        sys.exit("The valid ligands list (CSV file) do not exists inside the provided ligand xyz data folder: "+valid_ligs_file.as_posix())
    if not Path(valid_ligs_pc_file).is_file():
        sys.exit("The valid ligands list with images in qRank scale created (CSV file) do not exists inside the provided ligand xyz data folder: "+valid_ligs_pc_file.as_posix())

    # read csv with the valid ligands list to encode
    ligs_retrieve = pd.read_csv(valid_ligs_file, na_values = ['null', 'N/A'], keep_default_na = False) # do not interpret sodium NA as nan
    # filter out the low quality ligands
    ligs_retrieve = ligs_retrieve[ligs_retrieve.filter_quality]
    n = ligs_retrieve.shape[0]
    # sort by ligand Code, to test the same ligands together
    ligs_retrieve = ligs_retrieve.sort_values(['ligCode', 'missingHeavyAtoms']).reset_index(drop=True)

    # set grid_space information from the grid pc creation table
    ligs_retrieve_grid_space = pd.read_csv(valid_ligs_pc_file, na_values=['null', 'N/A'], keep_default_na=False,
                                           usecols=['ligID', 'grid_space'], nrows=1)
    grid_space = float(ligs_retrieve_grid_space.grid_space.values[0])
    del ligs_retrieve_grid_space
    elements_sphere_points = {
        reso: {e: 4 / 3 * np.pi * np.power(elements_radii_w_reso[reso][e], 3) / np.power(grid_space, 3)
               for e in elements_radii_w_reso[reso]}
        for reso in elements_radii_w_reso}

    # run the ligand pc qrank scale testing in multiprocessing
    # create the engine to test each ligand pc passing the parameters
    engine_testLigsPc = EngineTestLigandPointCloudQRankScale(db_ligxyz_path,pc_data_path,elements_sphere_points,
                                                               num_processors, draw_pc)
    ligands_out = engine_testLigsPc.run(ligs_retrieve)
    print("\nAverage time spent to test the "+str(ligands_out.shape[0])+
          " ligands point cloud: %.2f s." % ligands_out.time_process.mean())
    ligands_out = ligands_out.drop('time_process', axis=1)

    # save csv of the checked point clouds and print number of pdb and ligand errors
    ligs_retrieve = ligs_retrieve.merge(ligands_out, on='ligID')
    ligs_retrieve['grid_space'] = grid_space

    if ligs_retrieve.missing_data.any():
        print("- Missing ligands files:")
        print(str(ligs_retrieve.error[ligs_retrieve.missing_data]))

    error_ligands = ligs_retrieve.error.str.match("_pcLabels_wrongCount|_wrong_atoms_label|_pcLabels_error")
    if error_ligands.any():
        print("- Error ligands:")
        print(str(ligs_retrieve.error[error_ligands]))

    if ligs_retrieve.missing_data.any():
        print("A total of " + str(ligs_retrieve.missing_data.sum()) + "/", str(n),
              " ligands entries had at least one missing data.")

    if error_ligands.any():
        print("A total of "+str(error_ligands.sum())+"/", str(n),
              " ligands entries raised an error in the checking procedure.")

    ligs_retrieve = ligs_retrieve[ligs_retrieve.point_cloud]
    ligs_retrieve.to_csv(valid_ligs_file.as_posix().replace('.csv', '_qRankTested.csv'), index=False)

    d = time.time() - t0
    print("\nDONE!", "Time elapsed: %.2f s." % d)


if __name__ == "__main__":
    num_processors = 2
    draw_pc = False
    if len(sys.argv) >= 3:
        db_ligxyz_path = sys.argv[1]
        pc_data_path = sys.argv[2]
        if len(sys.argv) >= 4:
            num_processors = int(sys.argv[3])
        if len(sys.argv) >= 5:
            draw_pc = (sys.argv[4].lower() == "true")
    else:
        sys.exit("Wrong number of arguments. Two parameters must be supplied in order to test the image labeling of the "
                 "valid ligands and to compute two metrics for them: background class percentage e atoms coverage rate.\nParameters: \n"
                 "  1. xyz_labels_path: The path to the data folder called 'ligands/xyz_<valid ligand list csv name>' "
                 "where the ligands .xyz files with their atomic positions and structure labels are located. "
                 "It must also contain the CSV file with the valid ligands list and their grid sizing and position. "
                 "This file is named as '<valid ligand list csv name>_box_class_freq.csv' and is expected to be the "
                 "output of the 'run_vocabulary_encode_ligands.py' script. "
                 "Mandatory columns = 'ligID', 'ligCode', 'entry', 'filter_quality', 'x', 'y', 'z', 'x_bound', 'y_bound','z_bound';\n"
                 "  2. output_ligPCDB_path: The path to the data folder where the point clouds of the final images of the ligands in "
                 "quantile rank scale are stored ('data/lig_pcdb' or other);\n"
                 "  3. num_parallel: (optional) The number of processors to use for multiprocessing parallelization (default to 2);\n"
                 "  4. draw_pc: (optional) Boolean True or False to draw the images point clouds. "
                 "If True, enable drawing the final images of the ligands and color them using their labels. "
                 "If False, do not draw the images (default to False).\n"
                 )
    check_ligands_point_clouds(db_ligxyz_path, pc_data_path, num_processors, draw_pc)
