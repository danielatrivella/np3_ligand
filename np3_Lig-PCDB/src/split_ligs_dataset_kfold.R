library(anticlust)
library(readr)

# ligs_data_path <- "/home/crisfbazz/Documents/Unicamp/Mestrado/Projeto/np3_ligand/np3_pointcloud_DB/PDB_lists/ligands_valid_info_all_PDB_1.5_1.8_CHONPSIBrClFSe_atoms_free_ligands_1_counts_2008-02-01_depDate_filter_bfRatio_2.0_bfStd_10.0_occ_0.9_missHAtoms_True_numDisorder_0.0_box_class_freq_tested_pc.csv"
# ligs_data_path <- "data/ligands/xyz_test_ligands_valid_info_all_PDB_1.5_1.8_CHONPSIBrClFSe_atoms_free_ligands_1_counts_2008-02-01_depDate_filter_bfRatio_2.0_bfStd_10.0_occ_0.9_missHAtoms_True_numDisorder_0.0_box_class_freq/test_ligands_valid_info_all_PDB_1.5_1.8_CHONPSIBrClFSe_atoms_free_ligands_1_counts_2008-02-01_depDate_filter_bfRatio_2.0_bfStd_10.0_occ_0.9_missHAtoms_True_numDisorder_0.0_box_class_freq_box_class_freq_tested_pc.csv"
# vocab_path <- "/home/crisfbazz/Documents/Unicamp/Mestrado/Projeto/np3_ligand/np3_pointcloud_DB/PDB_lists/vocabulary_ligands_valid_info_all_PDB_1.5_1.8_CHONPSIBrClFSe_atoms_free_ligands_1_counts_2008-02-01_depDate_filter_bfRatio_2.0_bfStd_10.0_occ_0.9_missHAtoms_True_numDisorder_0.0.txt"
# kfolds <- 5
keep_labels <- "all"

# read input
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 3) {
  cat("Three parameters must be supplied to perform a stratified k-fold cross validation in the list of valid ligands. ",
      "It separates the entries in k similar groups of equal size and diverse characteristics. ",
      "Each k group is also separated in another two similar groups for test and validation subsets.\nParameters:\n",
      " 1. valid_ligands_list_path: Path to the CSV table with the list of valid ligands to be stratified with a ",
      "k-fold cross validation approach. Mandatory columns: ligCode, entry, bfactor, AverageBFactor, Resolution, ",
      "point_cloud_size_qRankMask, 0 to the number of classes - 1;\n",
      " 2. vocab_path: Path to the vocabulary file used to label the ligands entries present in the ",
      "valid_ligands_list_path table. It must contain one label by row, defining their order (the Background class is not used);\n",
      " 3. k: The number of anti-clusters (groups with high diversity) to be created. This is the number of k-folds. ",
      "Each k group will be separated in another two similar groups;\n",
      " 4. classes_list: (optional) The list of classes of the vocabulary that will be used in the separation of ",
      "the entries by the anti-clustering algorithm (stratified approach). The names of the selected classes ",
      "separated by comma or the word 'all' to use the entire vocabulary (all the classes). Default to 'all'.\n\n", sep="")
  stop("Wrong number of arguments.", call.=FALSE)
} else {
  ligs_data_path <- file.path(args[[1]])
  if (!file.exists(ligs_data_path))
  {
    stop("The ligand data file '", ligs_data_path,
         "' do not exists. Provide a valid path to where the csv file with the ligands ",
         "variables and entries is located.")
  }
  
  vocab_path <- file.path(args[[2]])
  if (!file.exists(vocab_path))
  {
    stop("The vocabulary file '", ligs_data_path,
         "' do not exists. Provide a valid path to where the csv file with the vocabulary ",
         "used to label the ligands is located.")
  }
  
  kfolds <- as.integer(args[[3]])
  if (is.na(kfolds))
  {
    stop("The provided number of anticlusters must be an integer. Wrong value provided.")
  }
  
  if (length(args) > 3)
  {
    keep_labels <- args[[4]]
  }
}

t0 <- Sys.time()

vocab <- read_lines(vocab_path)
num_classes <- length(vocab)
if (keep_labels != "all") {
  # split by comma
  keep_labels <- strsplit(keep_labels, ",")[[1]]
  # check if keep labels are present in the vocab
  if (!all(keep_labels %in% vocab)) {
    stop(paste("Wrong list of labels to be used in the ligands selection.",
               "Some of the provided labels are not present in the vocabulary: ",
               paste(keep_labels[!(keep_labels %in% vocab)], collapse = ",")))
  }
} else {
  keep_labels <- vocab
}
keep_labels <- match(keep_labels, vocab)-1


ligs_data <- read_csv(ligs_data_path)
# check if the vocab labels to keep and the mandatory columns names are present in the ligand data
if (!all(c("bfactor", "AverageBFactor", "Resolution", "ligCode", "entry",
           keep_labels) %in% 
         names(ligs_data)))
{
  stop("The ligand data file '", basename(ligs_data_path),
       "' do not have the mandatory columns. Provide a valid path to where the csv file with the ligands ",
       "variables and entries is located.")
}

pc_type <- c("point_cloud_size_qRankMask", "point_cloud_size_2sigmaMask")
if (pc_type[1] %in% names(ligs_data)) {
  pc_type <- pc_type[1]
} else if (pc_type[2] %in% names(ligs_data)) {
  pc_type <- pc_type[2]
} else {
  stop("The ligand data file '", basename(ligs_data_path),
       "' do not have the mandatory point cloud size column. Provide a valid path to where the csv file with the ligands ",
       "variables and entries is located.")
}

cat("** Start splitting the ligands data set with", nrow(ligs_data), "entries in",
    kfolds, "diverse groups **\n\n")


cat("\n\n* Classes by entry occurence ratio\n")
class_by_entry_ratio <- colSums(ligs_data[,format(keep_labels,trim = TRUE)]>0)
names(class_by_entry_ratio) <- vocab[as.numeric(keep_labels)+1]
print(class_by_entry_ratio)
cat("\n* Classes total occurence ratio\n")
class_total_ratio <- colSums(ligs_data[ ,format(keep_labels,trim = TRUE)])
names(class_total_ratio) <- vocab[as.numeric(keep_labels)+1]
print(class_total_ratio)

keep_labels <- keep_labels[class_by_entry_ratio > 0]
keep_labels <- format(keep_labels, trim = TRUE)

# compute the bfactor ratio between the ligand bfactor and the protein bfactor
ligs_data$bfactor_ratio <- ligs_data$bfactor / ligs_data$AverageBFactor
# set the features to be used in the anticlustering job
numeric_features <- c("bfactor_ratio", "Resolution", "min_occupancy", pc_type)
categorical_features <- c("ligCode", "entry")

cat("\n\n* Variables being used by the anticlustering algorithm:")
cat("\n* Numeric variables:\n")
cat(paste("- ",c(numeric_features,keep_labels), collapse = "\n"))
cat("\n* Categorical variables:\n")
cat(paste("- ",categorical_features, collapse = "\n"))

cat(paste("\n\n** Applying the anticlustering algorithm to separate the ligands ",
          "data set with", nrow(ligs_data), "entries in kfolds =", kfolds))
t1 <- Sys.time()
#ligs_data$kfolds <- anticlustering(ligs_data[,c(numeric_features,keep_labels)],
#                                   K = kfolds,
#                                   objective = "diversity",
#                                   method = "exchange",
#                                   categories = ligs_data[,categorical_features])
# if too much data use the following function
ligs_data$kfolds <- fast_anticlustering(ligs_data[,
                                                  c(numeric_features,keep_labels)],
                                        K =kfolds,
                                        categories = ligs_data[,
                                                               categorical_features])

t2 <- Sys.time()
cat("\nDone in",round(t2-t1, 2), units(t2-t1), "!\n\n")
print(by(ligs_data[, c(numeric_features,keep_labels)], 
         ligs_data$kfolds, function(x) round(colMeans(x), 2)))

cat("\n\n** Applying the anticlustering algorithm to equally separate each kfold set between test and validation")
t1 <- Sys.time()
ligs_data$test_val <- "test"
for (i in 1:kfolds) {
  sel_ligCode <- (ligs_data$kfolds == i)
  lig_anticlustering <- fast_anticlustering(ligs_data[sel_ligCode,
                                                      c(numeric_features,keep_labels)],
                                            K = 2,
                                            categories = ligs_data[sel_ligCode,
                                                                   categorical_features])
  ligs_data[sel_ligCode, "test_val"][lig_anticlustering == 2,] <- "val"
}
t2 <- Sys.time()
cat("\nDone in",round(t2-t1, 2), units(t2-t1), "!\n\n")

if (length(keep_labels) != num_classes) {
  # keep_labels <- paste0(vocab[as.numeric(keep_labels)+1], collapse = ",")
  keep_labels <- paste0(length(keep_labels), "classes")
} else {
  keep_labels <- "all"
}

write_csv(ligs_data, path=sub(x=ligs_data_path, pattern = ".csv", 
                              replacement = paste0("_split_",keep_labels,
                                                   "_kfolds_",kfolds,
                                                   ".csv")))
t2 <- Sys.time()
cat("\nFinish in",round(t2-t0, 2), units(t2-t0), "!\n\n")
# ligs_data[order(ligs_data$kfolds),
#           c("kfolds","ligCode", "entry", "bfactor_ratio", "Resolution")]
