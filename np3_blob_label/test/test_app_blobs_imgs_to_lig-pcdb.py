from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
from label_app_blobs_imgs import label_blob_imgs

# organized the ligands output images in folders following the hierarchic that is expected by the training pipeline
# and fixing the naming pattern. After copying the images, label them using .xyz labeled files for each ligand.
#
# the images will be organized inside subfolders named with the entryID name, inside a folder which is the new_output_path
# the entries_list table must have the columns "entryID" and "blobID"
# the blobID will be used to label the images in the new output directory
def organize_imgs_np3_blob_outs(entries_list_path, np3_output_path, new_output_path, db_ligxyz_path):
  np3_output_path = Path(np3_output_path)
  new_output_path = Path(new_output_path)
  entries_list_path = Path(entries_list_path)
  db_ligxyz_path = Path(db_ligxyz_path)

  # read the list of images to be moved
  entries_list = pd.read_csv(entries_list_path)

  # create the new output path if it does not exists yet
  new_output_path.mkdir(exist_ok=True, parents=True)
  n_correctly_cp = 0
  n_correctly_label = 0
  n_missing_img = 0
  for i in tqdm(range(entries_list.shape[0]), desc="Copy blobs imgs to new loc"):
    if i % 10 == 0:
      print("  Blob #", i, " ID", entries_list.blobID[i])
    # check if the respective blob image exists in the np3 output path
    blob_img_path = np3_output_path / entries_list.entryID[i] / "model_blobs" / "blobs_img" / entries_list.blobID[i] / \
                    str(entries_list.blobID[i] + "_point_cloud_fofc_qRankMask_5.xyzrgb")

    if blob_img_path.exists():
      # copy the blobs img to the new output location in the correct files organization
      new_blob_img_path = new_output_path / entries_list.entryID[i] / \
                          str(entries_list.blobID[i] + "_lig_point_cloud_fofc_qRankMask_5.xyzrgb")
      # create the new entry output path if it does not exist yet
      new_blob_img_path.parent.mkdir(exist_ok=True, parents=True)
      # copy blob image file
      try:
        shutil.copy(blob_img_path, new_blob_img_path)
        if i < 20:
          print("File copied successfully.")
        n_correctly_cp += 1

        # label copied blob img
        #print("entrou")
        n_correctly_label += label_blob_imgs(new_blob_img_path, db_ligxyz_path/(entries_list.blobID[i] + '_class.xyz'),
                                             entries_list.blobID[i], entries_list.Resolution[i])
      # If source and destination are same
      except shutil.SameFileError:
        if i < 20:
          print("Source and destination represents the same file.")
      # If there is any permission issue
      except PermissionError:
        if i < 20:
          print("Permission denied.")
      # For other errors
      except Exception as e:
        if i < 20:
          print("Error occurred while copying file.")
          print(e)
    else:
      n_missing_img += 1

  # print the final number of correctly copied files and misisng ones
  print("\nTotal of",n_missing_img, "/", entries_list.shape[0], "blobs image's were missing")
  print("Total of", n_correctly_cp, "/", entries_list.shape[0]-n_missing_img, "blobs image's successfully copied!!\n")
  print("Total of", n_correctly_label, "/", entries_list.shape[0]-n_missing_img,
        "blobs image's successfully labeled!!\n")


if __name__ == "__main__":
  import sys
  # parse arguments
  if len(sys.argv) >= 5:
    entries_list_path = sys.argv[1]
    np3_output_path = sys.argv[2]
    new_output_path = sys.argv[3]
    db_ligxyz_path = sys.argv[4]
  else:
    sys.exit("Wrong number of arguments. Three argument must be supplied to reorganize a NP3 Blob Label result into a LigPCDS"
             "dataset format. This script will copy the blobs point clouds to a new output directory"
             " following the input files naming and organization. "
             "Label the point clouds using the .xyz files from np3_LigPCDS scripts. \n"
             "  1. entries_list_path: The path to the CSV metadata table defining the ligands dataset that was used in the NP3 Blob Label process;\n"
             "  2. np3_ligand_output_path: The path to the output data folder where the np3 blob label result was stored for the "
             "desired job ('data/np3_ligand_<output_name>_<DATE>/');\n"
             "  3. new_output_path: the new output path to store the result;\n"
             "  4. db_ligxyz_path: The path to the xyz directory with the structure labels of the ligands present in the entries_list_path.\n"
             )
  organize_imgs_np3_blob_outs(entries_list_path, np3_output_path, new_output_path, db_ligxyz_path)


