# When we have an electron density map on a Grid we may want to check blobs – unmodelled electron density, potential ligand sites. Similarly to the “Unmodelled blobs” function in COOT, Gemmi has a function that finds such blobs. It was added to be used in CCP4 Dimple.
# Here we search for blobs in the difference electron density map, iterate over the blobs to extract their info and
# to add one fake atom at each returned blob in a new pdb to be used to facilitate the visualization of the results
import gemmi ## version 0.5.3
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time

def std_mean_median_q25th_q75th(x):
    return np.std(x), np.mean(x), np.median(x), np.quantile(x, 0.25), np.quantile(x, 0.75)

def gemmi_pos(pos):
    return gemmi.Position(pos[0], pos[1], pos[2])

def debug_view_blob_mask_img_fromGemmi(blob_mask):
        # code to visualization - for manual debug cases
        # input: return from gemmi.flood_fill_above function
        import open3d as o3d
        blob_points_index = np.where(blob_mask.array > 0)
        blob_positions = np.asarray([blob_mask.get_position(blob_points_index[0][i],blob_points_index[1][i],blob_points_index[2][i]).tolist() for i in range(len(blob_points_index[0]))])
        blob_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(blob_positions))
        o3d.visualization.draw_geometries([blob_pcd])

def retrieve_entry_density_map(entryID, refinement_data_path, grid_space, verbose):
    if verbose:
        logging.info("\n*** Start processing entry" + entryID + " electron density map ***")
    t1 = time.time()
    #
    mtz_file = refinement_data_path / (entryID + ".mtz")
    if not Path(mtz_file).is_file():
        if verbose:
            logging.info("Warning: The entry " + entryID +
                  " was not refined successfully, the mtz file is missing. All its blobs will be skipped.")
        return False

    # create the entry density map
    try:
        # read mtz file
        mtz = gemmi.read_mtz_file(mtz_file.as_posix())
        # mtz.nreflections   # from MTZ record NCOL
        # mtz.min_1_d2       # from RESO
        # mtz.max_1_d2       # from RESO
        # mtz.resolution_low()   # sqrt(1 / min_1_d2)
        # mtz.resolution_high() # sqrt(1 / max_1_d2)
        # mtz.datasets
        resolution = mtz.resolution_high()
        # check if map already exists before creating it again
        # and allow using a different map (entryID.ccp4) for the blob search and img creation
        if (refinement_data_path / (entryID + "_fofc.ccp4")).exists():
            if verbose:
                logging.info("* Already stored the entry Fo-Fc map in a ccp4 file. Skipping Fo-Fc map creation to blob's search. *")
            if (refinement_data_path / (entryID + ".ccp4")).exists():
                logging.info(
                    "* Another entry map named "+ (refinement_data_path / (entryID + ".ccp4")).name +
                    " was provided and will be used in the blob's search. *")
                map_grid_file = (refinement_data_path / (entryID + ".ccp4")).as_posix()
            else:
                map_grid_file = (refinement_data_path / (entryID + "_fofc.ccp4")).as_posix()
            map_grid = gemmi.read_ccp4_map(map_grid_file, setup=True).grid
            return map_grid, resolution
        #
        # define sample rate to interpolate density grid values
        # grid_space was informed
        sample_rate = mtz.resolution_high() / grid_space  # = dmin/grid_size

        if verbose:
            logging.info("Entry info:\nResolution high = " + str(mtz.resolution_high()) + "\nGrid space = " + str(grid_space) +
                  "\nSample rate = " + str(sample_rate))
        # mtz columns after REFMAC - https://www.globalphasing.com/buster/wiki/index.cgi?MTZcolumns
        # 2Fo-Fc map coefficients ->  amplitude: FWT pahses: PHWT
        # Fo-Fc (difference map) coefficients -> amplitude: DELFWT pahses: PHDELWT
        # Figure of merit -> FOM
        # Model structure factor -> FC_ALL PHIC_ALL : Atomic and bulk-solvent model;
        # Reflection indices -> H,K,L -> no scale
        # Observed data -> FP,SIGFP (or similar) : (amplitude and sigma). The column names will be identical to the column names of the input data -> scale Observational (unmodified from input)
        # Flag indicating which reflections are in the free set -> FreeR_flag : Copied from input if present, otherwise created afresh.
        # Calculated structure factors -> FC,PHIC
        #
        # obtain map 2Fo-Fc in direct space xyz
        # map_grid_2fofc = mtz.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=sample_rate)
        # obtain diff map Fo-Fc in direct space xyz
        # use 1.2 times the final sample rate to increase accuracy of the linear interpolation
        diff_map_col = 'DELFWT'
        diff_map_ph_col = 'PHDELWT'
        if not diff_map_col in mtz.column_labels() and not diff_map_ph_col in mtz.column_labels():
            diff_map_col = 'FOFCWT'
            diff_map_ph_col = 'PHFOFCWT'
            if not diff_map_col in mtz.column_labels() and not diff_map_ph_col in mtz.column_labels():
                logging.info("ERROR: no difference electron density map found in the mtz file: '" + mtz_file.as_posix()+
                         "'\nNo columns named as 'DELFWT' and 'PHDELWT' or as 'FOFCWT' and 'PHFOFCWT' were found.")
                sys.exit("ERROR: no difference electron density map found in the mtz file: '" + mtz_file.as_posix()+
                         "'\nNo columns named as 'DELFWT' and 'PHDELWT' or as 'FOFCWT' and 'PHFOFCWT' were found.")
        map_grid = mtz.transform_f_phi_to_map(diff_map_col, diff_map_ph_col,
                                                   sample_rate=sample_rate * 1.2)
        del mtz
        #
        # compute sigma contour
        # if verbose:
        #     logging.info("Computing the Electron Density map mean,std,median,quantile25th,quantile75th values")
        # # converting the map to a numpy array
        # p_std, p_mean, p_median, p_quantile25th, p_quantile75th = std_mean_median_q25th_q75th(np.array(map_grid,copy=False))

        # contour_sigma = [mean_I+std_I*-sigma_factor,mean_I+std_I*sigma_factor]
        # if verbose:
        #     logging.info("rho Standard deviation = "+ p_std + "\nrho mean = "+ p_mean + "\nrho median = "+ p_median +
        #           "\nrho quantile25th = " + p_quantile25th + "\nrho quantile75th = " + p_quantile75th)
        # "\nContour sigma = ", contour_sigma)
        #
        # apply symmetry if it is necessary - removed for now
        # map_grid_fofc.symmetrize_max()
        # std_I, mean_I = std_mean(np.array(map_grid_fofc, copy=False))
    except Exception as e:
        if verbose:
            logging.info("Error: Could not read and process the mtz file " + mtz_file.name + ". Skipping entry.")
            logging.exception(e)
        return False, None
    #
    # store the entry grid in a map file
    map_grid_fofc_file = (refinement_data_path / (entryID + "_fofc.ccp4")).as_posix()
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = map_grid
    ccp4.update_ccp4_header(2, True) # ccp4 file mode: 2 for floating-point data, 0 for masks ; // true to update min/max/mean/rms values in the header
    ccp4.write_ccp4_map(map_grid_fofc_file)
    if (refinement_data_path / (entryID + ".ccp4")).exists():
        logging.info(
            "* Another entry map named '" + (refinement_data_path / (entryID + ".ccp4")).name +
            "' was provided and will be used in the blob's search. *")
        map_grid_file = (refinement_data_path / (entryID + ".ccp4")).as_posix()
        map_grid = gemmi.read_ccp4_map(map_grid_file, setup=True).grid
    d = time.time() - t1
    if verbose:
        logging.info("Processed entry" + entryID + "in: %.2f s." % d)
    return map_grid, resolution

# search for blobs in specific positions
def search_blobs_parse_place_fake_atoms(blobs_list, entry_refinement_path, entry_output_path, grid_space,
                                        sigma_cutoff = 2.0, verbose=True):
    entry_refinement_path = Path(entry_refinement_path)
    entry_output_path = Path(entry_output_path)
    if not entry_output_path.exists() or not entry_output_path.is_dir():
        entry_output_path.mkdir(parents=True)

    entry_id = entry_refinement_path.name
    entry_mtz_file = entry_refinement_path / (entry_id+".mtz")
    entry_pdb_file = entry_refinement_path / (entry_id+".pdb")

    if not entry_mtz_file.exists() or not entry_pdb_file.exists():
        logging.info("WARNING: the mtz or pdb files of entry " + entry_id + " are missing. Skipping this entry!")
        return 0  # no blobs found due to missing entry

    # else resolve paths to get their absolute path
    entry_mtz_file = entry_mtz_file.resolve()
    entry_pdb_file = entry_pdb_file.resolve()

    grid, resolution = retrieve_entry_density_map(entry_id, entry_refinement_path, grid_space, verbose)
    if grid is False:
        logging.info("WARNING: the entry " + entry_id + " map grid could not be retrieved. Skipping this entry!")
        return 0  # no blobs found due to missing entry
    grid.normalize()

    # old - compute the sigma contour cutoff
    # points_mean_rho = np.mean(grid)  #grid.sum()/grid.point_count
    # points_std = np.std(grid)
    # rho_cutoff = points_mean_rho + points_std*sigma_cutoff
    # after normalization the cutoff is the sigma cutoff
    rho_cutoff = sigma_cutoff

    # read pdb
    st = gemmi.read_structure(entry_pdb_file.as_posix())
    # neighbor search with 8 A radius in the pdb
    # https://gemmi.readthedocs.io/en/latest/analysis.html?highlight=find%20residue%20location#neighbor-search
    ns = gemmi.NeighborSearch(st[0], st.cell, 8).populate()
    # add a new (empty) chain to store the fake atoms pointing to the blobs centroid
    new_chain = st[0].add_chain(gemmi.Chain('a'), unique_name=True)

    logging.info("*** Find blobs by flood fill around the given positions ***")
    logging.info("Parameters")
    logging.info(" - sigma cutoff:" + str(rho_cutoff))

    # blobs_list.loc[:,'blobScore'] = None
    # blobs_list.loc[:,'blobVolume'] = None
    # blobs_list.loc[:,'xyz_bound'] = None
    # blobs_list.loc[:,'chainResAtom'] = None
    # search for the blobs position
    logging.info("Blobs list:\n"+str(blobs_list)+ "\n")
    for i in range(blobs_list.shape[0]):
        if i < 3:
            logging.info("  Blob #"+ str(i+1))
        blob_pos = blobs_list.loc[i,['x','y','z']].values
        #blob_pos_gemmi = gemmi.Position(blob_pos[0],blob_pos[1],blob_pos[2])
        # blob_mask = gemmi.flood_fill_above(grid, [blob_pos_gemmi], threshold=rho_cutoff)
        # get the positions of the borders of a cube centered in the position to be searched
        # to prevent not finding blobs that are around the given position, but do not contain it
        cube_d = 1.0
        blob_center_cube_pos = [gemmi_pos(np.asarray(blob_pos) + adj) for adj in
                                [[0, 0, 0], [cube_d, 0, 0], [0, cube_d, 0], [0, 0, cube_d], [-cube_d, 0, 0], [0, -cube_d, 0],
                                 [0, 0, -cube_d]] +
                                [[x, y, z] for x in [cube_d, -cube_d] for y in [cube_d, -cube_d] for z in [cube_d, -cube_d]]]
        try:
            blob_mask = gemmi.flood_fill_above(grid, blob_center_cube_pos, threshold=rho_cutoff)
            blob_volume = blob_mask.sum() * grid.unit_cell.volume / grid.point_count
        except Exception as e:
            blob_volume = 0.0
            logging.info("Error searching for blob with flood_fill_above: " + str(e))

        # check if the flood fill was successful
        if blob_volume == 0:
            # search failed
            blobs_list.loc[i, 'blobVolume'] = 0
            blobs_list.loc[i, 'blobScore'] = -1
            blobs_list.loc[i, 'blobPeak'] = -1
            blobs_list.loc[i, 'blobMeanScore'] = -1
            blobs_list.loc[i, 'xyz_bound'] = 0
            blobs_list.loc[i, 'chainResAtom'] = None
            blobs_list.loc[i, 'chainResAtom_dist'] = -1
            logging.info("  - search failed :(")
        else:
            # get bounding box dimension from mask
            extent = blob_mask.get_nonzero_extent()  # bounding box containing the blob
            # logging.info("extent min: " + str(extent.minimum))
            # logging.info("extent max: " + str(extent.maximum))
            # bb_min = grid.unit_cell.orthogonalize(extent.minimum)
            # bb_max = grid.unit_cell.orthogonalize(extent.maximum)
            # logging.info("bb_min: " + str(bb_min.tolist()))
            # logging.info("bb_max: " + str(bb_max.tolist()))
            #bb dimension - abs diff
            # bb_dim = bb_max - bb_min
            # get the bb extent size from orthogonalized box
            bb_dim = grid.unit_cell.orthogonalize_box(extent).get_size()

            # very slow, not used -> alternative getting the extend directly from the positions plus a gap, but is slow
            # blob_points_index = np.where(blob_mask.array > 0)
            # blob_positions = np.asarray([blob_mask.get_position(blob_points_index[0][i], blob_points_index[1][i],
            #                                                     blob_points_index[2][i]).tolist() for i in
            #                              range(len(blob_points_index[0]))])
            # # [[blob_positions[:,0].min(),blob_positions[:,1].min(),blob_positions[:,2].min()], [blob_positions[:,0].max(),blob_positions[:,1].max(),blob_positions[:,2].max()]]
            # bb_min = np.asarray([blob_positions[:,0].min(), blob_positions[:,1].min(), blob_positions[:,2].min()])
            # bb_max = np.asarray([blob_positions[:,0].max(), blob_positions[:,1].max(), blob_positions[:,2].max()])
            # bb_dim = bb_max - bb_min
            # bb dimensions in x,y,z
            # xyz_bound = ','.join([str(round(abs(bb_i), 2)) for bb_i in bb_dim.tolist()])
            # add gap
            bb_gap = sigma_cutoff  # add a gap to include the entire blob
            xyz_bound = ','.join([str(round(abs(bb_i) + bb_gap, 2)) for bb_i in bb_dim])
            # bb center from the table, when orthogonalize the symmetric location is lost
            # bb_center = (bb_max + bb_min) / 2
            # blobs_list.loc[i, 'x'] = bb_center[0]
            # blobs_list.loc[i, 'y'] = bb_center[1]
            # blobs_list.loc[i, 'z'] = bb_center[2]
            # compute blob score by summing non zero values
            blob_score = grid.array[np.where(blob_mask.array > 0)].sum()
            # compute the peak intensity by getting the maximum value in the blob mask
            blob_peak = grid.array[np.where(blob_mask.array > 0)].max()
            blobs_list.loc[i, 'blobVolume'] = blob_volume
            blobs_list.loc[i, 'blobScore'] = blob_score
            blobs_list.loc[i, 'blobPeak'] = blob_peak
            blobs_list.loc[i, 'blobMeanScore'] = blob_score / blob_volume
            blobs_list.loc[i, 'xyz_bound'] = xyz_bound  # old ->np.ceil((blob_volume*3/4/np.pi)**(1/3) * 2 * 2 * 1.5)

            # search the closer residue to the blob and then assign its centroid as a fake atom to a new fake chain
            # if nothing found, append None
            cra = ns.find_nearest_atom(gemmi_pos(blob_pos))
            cra_dist = -1
            if cra is not None:
                cra_dist = cra.pos().dist(gemmi_pos(blob_pos))
                cra = cra.to_cra(st[0])
            # set near atom information: chain and residue location and distance to the blob center
            blobs_list.loc[i, 'chainResAtom'] = str(cra)
            blobs_list.loc[i, 'chainResAtom_dist'] = cra_dist
            # st[0][cra.chain.name][str(cra.residue.seqid.num)][cra.residue.name]
            # create a fake atom positioned in the blob centroid and assign it to a fake residue
            new_atom = gemmi.Atom()
            new_atom.name = ' B' + str(i+1)
            new_atom.flag = 'H'
            new_atom.pos = gemmi.Position(blob_pos[0], blob_pos[1], blob_pos[2])
            new_atom.b_iso = blobs_list.loc[i, 'blobScore']
            # create a fake residue to store the atom pointing to the blob and then add it to the new chain storing all blobs centers
            new_res = gemmi.Residue()
            new_res.name = "BLB"
            new_atom = new_res.add_atom(new_atom)
            new_res.het_flag = 'H'
            new_res.seqid = gemmi.SeqId(str(i + 1))
            # add residue with blob to the new_chain
            new_res = new_chain.add_residue(new_res)

            # print info for the first 3 blobs
            if i < 3:
                logging.info("  - given position: " + str(blob_pos))
                logging.info("  - score: "+ str(blobs_list.loc[i, 'blobScore']))
                logging.info("  - peak value: " + str(blobs_list.loc[i, 'blobPeak']))
                logging.info("  - volume: " + str(blobs_list.loc[i, 'blobVolume']))
                #logging.info("  - diameter: " + str(np.round((blobs_list.loc[i, 'blobVolume'] * 3 / 4 / np.pi) ** (1 / 3) * 2, 2)) + " A")
                logging.info("  - xyz bounding box dimensions: " + str(xyz_bound) + " A")  # np.ceil((blobs_list.loc[i, 'blobVolume'] * 3 / 4 / np.pi) ** (1 / 3) * 2 * 2 * 1.5), "A")

    n_blobs = (blobs_list.blobVolume>0).sum()
    if n_blobs > 0:
        # write the pdb with the fake atoms pointing to the blobs centroid, to be used in the visualization of the results
        st.write_minimal_pdb(entry_output_path.as_posix() + "/" + entry_pdb_file.name.replace('.pdb', '_blobs.pdb'))
        # set the entry resolution and the used grid_space
        blobs_list["resolution"] = resolution
        blobs_list["grid_space"] = grid_space
        # remove refinement columns
        blobs_list.drop(["refinement", "noHetatm"], axis=1, inplace=True)
        # write the blobs list to a csv, quote text
        blobs_list.to_csv(entry_output_path / 'blobs_list.csv', index=False, quoting=2)

        # create a Coot python script to open the entry pdb and mtz file and serve as a template for the blobs visualization script
        entry_coot_view = "#!/usr/bin/env coot\n" \
                          "# python script for coot - generated by np3_ligand inspired by dimple\n" \
                          "set_nomenclature_errors_on_read('ignore')\n" \
                          "molecule = read_pdb('"+entry_pdb_file.as_posix()+"')\n" \
                          "set_rotation_centre("+str(blobs_list[blobs_list.blobVolume>0].x.values[0])+", "+\
                          str(blobs_list[blobs_list.blobVolume>0].y.values[0])+", "+\
                          str(blobs_list[blobs_list.blobVolume>0].z.values[0])+")\n" \
                          "set_zoom(30.)\n" \
                          "refl = '"+entry_mtz_file.as_posix()+"'\n" \
                          "map21 = make_and_draw_map(refl, 'FWT', 'PHWT', '', 0, 0)\n" \
                          "if map21 == -1:\n" \
                          "    map21 = make_and_draw_map(refl, '2FOFCWT', 'PH2FOFCWT', '', 0, 0)\n" \
                          "map11 = make_and_draw_map(refl, 'DELFWT', 'PHDELWT', '', 0, 1)\n" \
                          "if map11 == -1:\n" \
                          "    map11 = make_and_draw_map(refl, 'FOFCWT', 'PHFOFCWT', '', 0, 1)\n"
        if (entry_refinement_path / (entry_id + ".ccp4")).exists():
            # if a ccp4 map was provided and used in the search for blobs and img creation,
            # also open it in the Coot script
            entry_map_provided = (entry_refinement_path / (entry_id + ".ccp4")).resolve()
            entry_coot_view = entry_coot_view + "handle_read_ccp4_map('" + entry_map_provided.as_posix() + "', 1)\n"
            entry_coot_view = entry_coot_view + "set_last_map_contour_level_by_sigma(2)\n"
            entry_coot_view = entry_coot_view + "set_last_map_color(0.588,1,0)\n"  # color purple

        entry_coot_view_path = entry_refinement_path / "entry-view-coot.py"
        entry_coot_view_path.write_text(entry_coot_view)
    else:
        logging.info("No blob was found in the given positions! =/" +
              "\nTry again with a smaller cutoff.")
    # return number of blobs that were found
    return n_blobs


# find all blobs in a given entry
def find_blobs_parse_place_fake_atoms(entry_refinement_path, entry_output_path, grid_space,
                                      sigma_cutoff = 3.0, blob_min_volume = 20, blob_min_score = 10, blob_min_peak = 0,
                                      verbose = True):
    entry_refinement_path = Path(entry_refinement_path)
    entry_output_path = Path(entry_output_path)
    if not entry_output_path.exists() or not entry_output_path.is_dir():
        entry_output_path.mkdir(parents=True)

    entry_id = entry_refinement_path.name
    entry_mtz_file = entry_refinement_path / (entry_id+".mtz")
    entry_pdb_file = entry_refinement_path / (entry_id+".pdb")
    if not entry_mtz_file.exists() or not entry_pdb_file.exists():
        logging.info("WARNING: the mtz or pdb files of entry " + entry_id + " are missing. Skipping this entry!")
        return 0  # no blobs found due to missing entry

    # else resolve paths to get their absolute path
    entry_mtz_file = entry_mtz_file.resolve()
    entry_pdb_file = entry_pdb_file.resolve()

    # load entry map: a provided one or from MTZ fo-fc
    grid, resolution = retrieve_entry_density_map(entry_id, entry_refinement_path, grid_space, verbose)
    if grid is False:
        logging.info("WARNING: the entry " + entry_id + " map grid could not be retrieved. Skipping this entry!")
        return 0  # no blobs found due to missing entry
    grid.normalize()

    # old - compute the sigma contour cutoff and the minimum blob volume
    # points_mean_rho = grid.sum()/grid.point_count
    # points_std = np.std(grid)
    # rho_cutoff = points_mean_rho + points_std*sigma_cutoff
    # after normalization the cutoff is the sigma cutoff
    rho_cutoff = sigma_cutoff
    # default min volume equals 4/3*np.pi*1.0**3 * 5 atoms ~ 20 - equivalent to 5 atoms with radius equals 1

    logging.info("*** Find blobs by flood fill ***")
    logging.info("Parameters")
    logging.info(" - sigma cutoff: " + str(rho_cutoff))
    logging.info(" - min_volume: " + str(blob_min_volume))
    logging.info(" - min_score: " + str(blob_min_score))
    logging.info(" - min_peak:" + str(blob_min_peak))

    # find the blob
    # sort by volume
    blobs = gemmi.find_blobs_by_flood_fill(grid, cutoff=rho_cutoff, min_volume=blob_min_volume, min_score=blob_min_score,
                                           min_peak=blob_min_peak)
    blobs.sort(key=lambda a: a.volume, reverse=True)
    n_blobs = len(blobs)

    if n_blobs > 0:
        # read pdb
        st = gemmi.read_structure(entry_pdb_file.as_posix())
        # neighbor search with 8 A radius in the pdb
        #https://gemmi.readthedocs.io/en/latest/analysis.html?highlight=find%20residue%20location#neighbor-search
        ns = gemmi.NeighborSearch(st[0], st.cell, 8).populate()
        # add a new (empty) chain to store the fake atoms pointing to the blobs centroid
        new_chain = st[0].add_chain(gemmi.Chain('a'), unique_name=True)

        # iterate over the blobs to extract their info and to add one fake atom at each returned blob
        # use neighbor search to find #chain #residue to add a fake atom
        blobs_list = {'entryID': [], 'blobID': [], 'x': [], 'y': [], 'z': [],  'blobVolume': [],
                      'blobScore': [], 'blobPeak': [], 'blobMeanScore': [], 'xyz_bound': [], 'chainResAtom':[],
                      'chainResAtom_dist':[]}
        logging.info("\n* Blobs list *\n")
        for i, blob in enumerate(blobs):
            if i < 3:
                logging.info("  Blob #" + str(i+1))

            # get mask and bounding box dimension
            seed = blob.peak_pos
            blob_mask = gemmi.flood_fill_above(grid, [seed], threshold=rho_cutoff)  # bounding box containing the blob
            extent = blob_mask.get_nonzero_extent()
            # bb_min = grid.unit_cell.orthogonalize(extent.minimum)
            # bb_max = grid.unit_cell.orthogonalize(extent.maximum)
            # # bb dimension - abs diff
            # bb_dim = bb_max - bb_min
            # get the bb extent size from orthogonalized box
            bb_dim = grid.unit_cell.orthogonalize_box(extent).get_size()
            # testing getting the extend directly from the positions plus a gap
            # blob_points_index = np.where(blob_mask.array > 0)
            # blob_positions = np.asarray([blob_mask.get_position(blob_points_index[0][i], blob_points_index[1][i],
            #                                                     blob_points_index[2][i]).tolist() for i in
            #                              range(len(blob_points_index[0]))])
            # # [[blob_positions[:,0].min(),blob_positions[:,1].min(),blob_positions[:,2].min()], [blob_positions[:,0].max(),blob_positions[:,1].max(),blob_positions[:,2].max()]]
            # bb_min = np.asarray([blob_positions[:, 0].min(), blob_positions[:, 1].min(), blob_positions[:, 2].min()])
            # bb_max = np.asarray([blob_positions[:, 0].max(), blob_positions[:, 1].max(), blob_positions[:, 2].max()])
            # bb_dim = bb_max - bb_min
            # bb dimensions in x,y,z
            bb_gap = sigma_cutoff  # add a gap to include the entire blob
            xyz_bound = ','.join([str(round(abs(bb_i) + bb_gap,2)) for bb_i in bb_dim])  #.tolist()])
            # bb center
            bb_center = blob.centroid  # (bb_max + bb_min)/2
            #
            blobs_list['entryID'].append(entry_id)
            blobs_list['blobID'].append(entry_id+'_blob_'+str(i+1))
            blobs_list['x'].append(bb_center[0])
            blobs_list['y'].append(bb_center[1])
            blobs_list['z'].append(bb_center[2])
            blobs_list['blobVolume'].append(blob.volume)
            blobs_list['blobScore'].append(blob.score)
            blobs_list['blobPeak'].append(blob.peak_value)
            blobs_list['blobMeanScore'].append(blob.score/blob.volume)
            #
            # compute the grid bounding box size by considering the volume as a sphere to get its radius,
            # then multiple by 2 to get the diameter and expand it by 300%
            blobs_list['xyz_bound'].append(xyz_bound)  # old -> np.ceil((blob.volume*3/4/np.pi)**(1/3) * 2 * 2 * 1.5))

            if i < 3:
                logging.info("  - centroid " + str(bb_center))
                logging.info("  - peak pos " + str(blob.peak_pos))
                logging.info("  - peak value " + str(np.round(blob.peak_value,2)))
                logging.info("  - score " + str(np.round(blob.score,2)))
                logging.info("  - volume " + str(np.round(blob.volume,2)) + " A^3")
                logging.info("  - xyz bounding box dimensions " + str(xyz_bound) + " A")  #np.ceil((blob.volume*3/4/np.pi)**(1/3) * 2 * 2 * 1.5), "A")
            # search the closer residue to the blob and then assign its centroid as a fake atom to a new fake chain
            # if nothing found, append None
            cra = ns.find_nearest_atom(blob.centroid)
            cra_dist = -1
            if cra is not None:
                cra_dist = cra.pos().dist(blob.centroid)
                cra = cra.to_cra(st[0])
            # set near atom information: chain and residue location and distance to the blob center
            blobs_list['chainResAtom'].append(str(cra))
            blobs_list['chainResAtom_dist'].append(cra_dist)
            # st[0][cra.chain.name][str(cra.residue.seqid.num)][cra.residue.name]
            # create a fake atom positioned in the blob centroid and assign it to a fake residue
            new_atom = gemmi.Atom()
            new_atom.name = ' B' + str(i+1)
            new_atom.flag = 'H'
            new_atom.pos = gemmi.Position(blob.centroid[0], blob.centroid[1], blob.centroid[2])
            new_atom.b_iso = blob.score
            # create a fake residue to store the atom pointing to the blob and then add it to the new chain storing all blobs centers
            new_res = gemmi.Residue()
            new_res.name = "BLB"
            new_atom = new_res.add_atom(new_atom)
            new_res.het_flag = 'H'
            new_res.seqid = gemmi.SeqId(str(i + 1))
            # add residue with blob to the new_chain
            new_res = new_chain.add_residue(new_res)

        del grid
        # write the pdb with the fake atoms pointing to the blobs centroid, to be used in the visualization of the results
        st.write_minimal_pdb(entry_output_path.as_posix() + "/" + entry_pdb_file.name.replace('.pdb', '_blobs.pdb'))
        # write the blobs list to a csv, quote text
        blobs_table = pd.DataFrame(blobs_list)
        # set the entry resolution and the used grid_space
        blobs_table["resolution"] = resolution
        blobs_table["grid_space"] = grid_space
        blobs_table.to_csv(entry_output_path / 'blobs_list.csv', index=False, quoting=2)

        # create a Coot python script to open the entry pdb and mtz file and serve as a template for the blobs visualization script
        entry_coot_view = "#!/usr/bin/env coot\n" \
                          "# python script for coot - generated by np3_ligand inspired by dimple\n" \
                          "set_nomenclature_errors_on_read('ignore')\n" \
                          "molecule = read_pdb('"+entry_pdb_file.as_posix()+"')\n" \
                          "set_rotation_centre("+str(blobs_table.x[0])+", "+str(blobs_table.y[0])+", "+str(blobs_table.z[0])+")\n" \
                          "set_zoom(30.)\n" \
                          "refl = '"+entry_mtz_file.as_posix()+"'\n" \
                          "map21 = make_and_draw_map(refl, 'FWT', 'PHWT', '', 0, 0)\n" \
                          "if map21 == -1:\n" \
                          "    map21 = make_and_draw_map(refl, '2FOFCWT', 'PH2FOFCWT', '', 0, 0)\n" \
                          "map11 = make_and_draw_map(refl, 'DELFWT', 'PHDELWT', '', 0, 1)\n" \
                          "if map11 == -1:\n" \
                          "    map11 = make_and_draw_map(refl, 'FOFCWT', 'PHFOFCWT', '', 0, 1)\n"
        if (entry_refinement_path / (entry_id + ".ccp4")).exists():
            # if a ccp4 map was provided and used in the search for blobs and img creation,
            # also open it in the Coot script
            entry_map_provided = (entry_refinement_path / (entry_id + ".ccp4")).resolve()
            entry_coot_view = entry_coot_view + "handle_read_ccp4_map('" + entry_map_provided.as_posix() + "', 1)\n"
            entry_coot_view = entry_coot_view + "set_last_map_contour_level_by_sigma(2)\n"
            entry_coot_view = entry_coot_view + "set_last_map_color(0.588,1,0)\n"  # color purple
        entry_coot_view_path = entry_refinement_path / "entry-view-coot.py"
        entry_coot_view_path.write_text(entry_coot_view)
    else:
        logging.info("No blob was found!")
    # return number of blobs that were found
    return n_blobs



if __name__ == "__main__":
    # read the entry mtz and pdb, find blos, save blobs list, create fake atoms poiting to the the blobs location
    sigma_cutoff = 3
    grid_space=0.5
    blob_min_volume = 20.0
    blob_min_score = 10
    blob_min_peak = 0
    # parse arguments
    if len(sys.argv) >= 3:
        entry_refinement_path = sys.argv[1]
        entry_output_path = sys.argv[2]
        if len(sys.argv) >= 4:
            grid_space = float(sys.argv[3])
        if len(sys.argv) >= 5:
            sigma_cutoff = float(sys.argv[4])
        if len(sys.argv) >= 6:
            blob_min_volume = float(sys.argv[5])
        if len(sys.argv) >= 7:
            blob_min_score = float(sys.argv[6])
        if len(sys.argv) >= 8:
            blob_min_peak = float(sys.argv[7])
    else:
        sys.exit("Wrong number of arguments. Two argument must be supplied in order to read an entry refinement "
                 "path and search for blobs using the entry mtz and pdb files or a provided ccp4 map for that entry: \n"
                 "  1. The path to the data folder where the entry refinement is located ('data/refinement/entryID'). "
                 "It must contain the pdb and mtz files resulting from the refinement process (e.g. dimple result), "
                 "named as 'entryID.pdb' and 'entryID.mtz'. "
                 "Optionally it may contain a 'entryID.ccp4' map to be used in the blobs search. "
                 "A table named 'blobs_list.csv' will be created in this folder and a pdb file will also be created named "
                 "'entryID_blobs.pdb' with the blobs location added as fake atoms in a fake chain of this structure to "
                 "be used in the results visualization;\n"
                 "  2. The path to the output data folder where the np3 ligand result is being stored for the current"
                 "entry ('data/np3_ligand_<DATE>/entryID');\n"
                 "  3. (optional) grid_space: The grid space size in angstroms to be used to create the point clouds or"
                 " FALSE when informing the sample_rate (Default to 0.5 A). It should be the same of the used model.\n"
                 "  4. (optional) sigma_cutoff: A numeric defining the sigma cutoff to be used to search for blobs in "
                 "the difference electron density map (Default to 2.5);\n"
                 "  5. (optional) blob_min_volume: A numeric defining the minimum volume that a blob must have to be "
                 "considered. Default to 20 A^3, what is equivalent to a molecule with about 5 atoms;\n"
                 "  6. (optional) blob_min_score: A numeric defining the minimum score that a blob must have to be "
                 "considered (Default to 0);\n"
                 "  7. (optional) blob_min_peak: A numeric defining the minimum intensity that the peak of a blob "
                 "must have to be considered (Default to 0).\n"
                 )
    find_blobs_parse_place_fake_atoms(entry_refinement_path, entry_output_path, grid_space, sigma_cutoff=sigma_cutoff,
                                      blob_min_volume=blob_min_volume, blob_min_score=blob_min_score,
                                      blob_min_peak=blob_min_peak)

