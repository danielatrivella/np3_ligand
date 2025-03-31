library(anticlust)
library(readr)

# ligs_data_path <- "/home/crisfbazz/Documents/Unicamp/Mestrado/Projeto/np3_ligand/np3_pointcloud_DB/PDB_lists/ligands_valid_info_all_PDB_1.5_1.8_CHONPSIBrClFSe_atoms_free_ligands_1_counts_2008-02-01_depDate_filter_bfRatio_2.0_bfStd_10.0_occ_0.9_missHAtoms_True_numDisorder_0.0_box_class_freq_tested_pc.csv"
# ligs_data_path <- "data/ligands/xyz_test_ligands_valid_info_all_PDB_1.5_1.8_CHONPSIBrClFSe_atoms_free_ligands_1_counts_2008-02-01_depDate_filter_bfRatio_2.0_bfStd_10.0_occ_0.9_missHAtoms_True_numDisorder_0.0_box_class_freq/test_ligands_valid_info_all_PDB_1.5_1.8_CHONPSIBrClFSe_atoms_free_ligands_1_counts_2008-02-01_depDate_filter_bfRatio_2.0_bfStd_10.0_occ_0.9_missHAtoms_True_numDisorder_0.0_box_class_freq_box_class_freq_tested_pc.csv"
# vocab_path <- "/home/crisfbazz/Documents/Unicamp/Mestrado/Projeto/np3_ligand/np3_pointcloud_DB/PDB_lists/vocabulary_ligands_valid_info_all_PDB_1.5_1.8_CHONPSIBrClFSe_atoms_free_ligands_1_counts_2008-02-01_depDate_filter_bfRatio_2.0_bfStd_10.0_occ_0.9_missHAtoms_True_numDisorder_0.0.txt"
# keep_labels <- "all" # "sp2CA6,sp2,sp3"
# max_num_ligCode <- 500
# min_class_occ <- 5000
# max_class_occ <- 10000

# read input
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 6) {
  cat("Six parameters must be supplied to undersample the given list of valid ligands (dataset) using the ligands",
       "entries occurrence by class (within the selected ones) and by ligand code (unique structure).\nParameters:\n",
       " 1. valid_ligands_list_path: Path to the CSV table with the list of valid ligands. ",
       "The undersampling technique will be applied in this list to filter the ligands entries (rows), ",
       "it will remove bias towards frequent ligand codes and frequent classes. ",
       "Mandatory columns: ligCode, entry, bfactor, AverageBFactor, Resolution, point_cloud_size_qRank0.95, and '0' to the number of classes in the vocabulary minus one;\n",
       " 2. vocab_path: Path to the vocabulary file used to label the ligands entries present in the ",
       "valid_ligands_list_path table. It must contain one label by row, defining their order (the Background class is not used);\n",
       " 3. classes_list: The list of classes of the vocabulary that will be used in the undersampling of the ",
       "entries by the anti-clustering algorithm (stratified approach). ",
       "The names of the selected classes separated by comma or the word 'all' to use the entire vocabulary (all the classes). ",
       "Only the ligands that were labeled with this list of classes will be kept, the rest will be filtered out;\n",
       " 4. max_ligCode_occ: The maximum number of ligand entries occurrences by ligCode - ",
       "balance the occurrence of different ligands structures in the dataset;\n",
       " 5. min_class_occ: The minimum number of classes occurrences by ligand entry (minimum number of entries in which the class appear);\n",
       " 6. max_class_occ: The maximum number of classes occurrences by ligand entry (maximum number of entries in which the class appear).\n\n",
       sep="")
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
  
  keep_labels <- args[[3]]
  
  max_num_ligCode  <- as.integer(args[[4]])
  if (is.na(max_num_ligCode))
  {
    stop("The provided maximum number of ligand occurrences by ligCode must be an integer. Wrong value provided.")
  }
  
  min_class_occ  <- as.integer(args[[5]])
  if (is.na(min_class_occ))
  {
    stop("The provided minimum number of classes occurrences by ligand entry. Wrong value provided.")
  }
  
  max_class_occ  <- as.integer(args[[6]])
  if (is.na(max_class_occ))
  {
    stop("The provided maximum number of lclasses occurrences by ligand entry. Wrong value provided.")
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
if (!all(c("bfactor", "AverageBFactor", "Resolution", "ligCode", "entry", keep_labels) %in% 
         names(ligs_data)))
{
  stop("The ligand data file '", basename(ligs_data_path),
       "' do not have the mandatory columns. Provide a valid path to where the csv file with the ligands ",
       "variables and entries is located.")
}

pc_type <- c("point_cloud_size_qRank0.95", "point_cloud_size_3sigma")
if (pc_type[1] %in% names(ligs_data)) {
  pc_type <- pc_type[1]
} else if (pc_type[2] %in% names(ligs_data)) {
  pc_type <- pc_type[2]
} else {
  stop("The ligand data file '", basename(ligs_data_path),
       "' do not have the mandatory point cloud size column. Provide a valid path to where the csv file with the ligands ",
       "variables and entries is located.")
}

cat("\n** Start undersampling the ligands data set with", nrow(ligs_data), 
    "entries  **\n")
cat("- keep_labels: ",keep_labels, "\n- max_num_ligCode: ",max_num_ligCode,
    "\n- min_class_occ: ",min_class_occ, "\n- max_class_occ: ",max_class_occ,"\n\n")

# filter the ligands that only have the labels present in the keep_labels list
if (length(keep_labels) != num_classes) {
  cat("* Filter the ligands that contains the labels present in the provided labels list:",keep_labels,"\n")
  vocab_cols_rm <- 0:(num_classes-1)
  vocab_cols_rm <- format(vocab_cols_rm[-(keep_labels+1)], trim = TRUE)
  keep_labels <- format(keep_labels, trim = TRUE)
  rm_ligs_by_labels <- (rowSums(ligs_data[,keep_labels]) > 0 & 
                          rowSums(ligs_data[,vocab_cols_rm]) == 0)
  ligs_data <- ligs_data[rm_ligs_by_labels, ]
  # print how many ligands were excluded
  cat(paste("\n# Removed a total of", sum(!rm_ligs_by_labels), 
            "ligands that have labels different from the following list:", 
            paste0(vocab[as.numeric(keep_labels)+1], collapse = ","),
            ".\n"))
} else {
  keep_labels <- format(keep_labels, trim = TRUE)
}

# ligCodes to be removed, by default remove the ligands with only one atom different from H
rm_ligCode_list <- c("BR", "CL", "F", "IOD", "OH", "H2S", "O", "NH2", "NH3", "NH4")
cat("* The following list of ligands codes with only one atom different from H will be removed:", rm_ligCode_list)
rm_ligCode <- (ligs_data$ligCode %in% rm_ligCode_list)
if (sum(rm_ligCode) > 0) {
  cat(paste("\n- Removed a total of", sum(rm_ligCode), "ligands with only one atom",
            "different from H.\n"))
  ligs_data <- ligs_data[!rm_ligCode,]
}

cat("\n\n* Filter out the ligands entries with a point cloud size at 0.95 qRank smaller than 150 points.")
rm_ligs_by_size <- (ligs_data[[pc_type]] >= 150)
# filter ligs with pc_type > 150
ligs_data <- ligs_data[rm_ligs_by_size, ]
# print how many ligands were excluded
cat("\n- Removed a total of", sum(!rm_ligs_by_size), 
    "ligands that have a point cloud size with less than 150 points.")

cat("\n\n* Entries by class occurrence ratio\n")
class_by_entry_ratio <- colSums(ligs_data[,keep_labels]>0)
names(class_by_entry_ratio) <- vocab[as.numeric(keep_labels)+1]
print(class_by_entry_ratio)
cat("\n* Total atoms by class occurrence ratio\n")
class_total_ratio <- colSums(ligs_data[ ,keep_labels])
names(class_total_ratio) <- vocab[as.numeric(keep_labels)+1]
print(class_total_ratio)

# filter the classes using the min occurrence by class
rm_ligs_by_labels <- c(1)
while(sum(rm_ligs_by_labels) > 0) {
  # labels_occ = colSums(ligs_data[,keep_labels])
  # names(labels_occ) <- vocab[as.numeric(keep_labels)+1]
  class_by_entry_ratio <- colSums(ligs_data[,keep_labels]>0)
  names(class_by_entry_ratio) <- vocab[as.numeric(keep_labels)+1]
  exclude_labels <- format(match(names(class_by_entry_ratio[class_by_entry_ratio < min_class_occ]), vocab)-1,
                           trim = TRUE)
  if (length(exclude_labels) == 0) { break() }
  cat("\n* Filtering the ligands entries that contains a class with less than the minimum occurrence threshold ")
  cat("\n    The ligands with the following labels will be removed:", paste(vocab[as.numeric(exclude_labels)+1], collapse = ","))
  rm_ligs_by_labels <- (rowSums(ligs_data[,exclude_labels]) > 0)
  cat("\n- Removed a total of", sum(rm_ligs_by_labels), 
      "ligands")
  ligs_data <- ligs_data[!rm_ligs_by_labels,]
  class_by_entry_ratio <- colSums(ligs_data[,keep_labels]>0)
  keep_labels <- keep_labels[class_by_entry_ratio > 0]
}
# print(rm_ligs_by_labels)
if (sum(rm_ligs_by_labels) != 1) {
  cat("\n\n* Entries by class occurrence ratio\n")
  class_by_entry_ratio <- colSums(ligs_data[,keep_labels]>0)
  names(class_by_entry_ratio) <- vocab[as.numeric(keep_labels)+1]
  print(class_by_entry_ratio)
  cat("\n* Total atoms by class occurrence ratio\n")
  class_total_ratio <- colSums(ligs_data[ ,keep_labels])
  names(class_total_ratio) <- vocab[as.numeric(keep_labels)+1]
  print(class_total_ratio)
}

# compute the bfactor ratio between the ligand bfactor and the protein bfactor
ligs_data$bfactor_ratio <- ligs_data$bfactor / ligs_data$AverageBFactor
# sort the ligs data using the lig code and the bfactor to equally distribute the ligands entries using this two variables
ligs_data <- ligs_data[order(ligs_data$ligCode, ligs_data$bfactor_ratio),]
# set the features to be used in the anticlustering job
numeric_features <- c("bfactor_ratio", "Resolution", "min_occupancy", pc_type)
categorical_features <- c("ligCode", "entry")

cat("\n\n* Variables being used by the anticlustering algorithm:")
cat("\n* Numeric variables:\n")
cat(paste("- ",c(numeric_features,keep_labels), collapse = "\n"))
cat("\n* Categorical variables:\n")
cat(paste("- ",categorical_features, collapse = "\n"))

# add parameter of maximum number of each ligCode, exclude using anticlustering lig_kfolds print the ones that most appears
total_ligCode = table(ligs_data$ligCode)
total_ligCode = total_ligCode[total_ligCode > max_num_ligCode]
if (length(total_ligCode) > 0) {
  #table(ligs_data$ligCode)
  t1 <- Sys.time()
  cat(paste("\n\n* The following ligands code have more than", max_num_ligCode,
            "occurrences and their surplus will be removed using the anticlustering algorithm\n"))
  print(total_ligCode)
  for (ligCode in names(total_ligCode)) {
    cat("\n* Limiting ligand ", ligCode)
    kfolds_lig <- ceiling(total_ligCode[ligCode]/(max_num_ligCode))
    if (kfolds_lig == 1) {
      kfolds_lig <- kfolds_lig + 1
    }
    sel_ligCode <- (ligs_data$ligCode == ligCode)
    lig_anticlustering <- fast_anticlustering(ligs_data[sel_ligCode,
                                                        c(numeric_features,keep_labels)],
                                              K = kfolds_lig,
                                              k_neighbours = max_num_ligCode,
                                              categories = ligs_data[sel_ligCode,
                                                                     "entry"])
    # keep the first anticlustering groups -> equivalent to approx max_num_ligCode ligands
    sel_rm <- which(sel_ligCode)[lig_anticlustering > 1]
    # if keeping less ligands than the maximum, sample the remaining to achieve the max num of ligCodes
    if (total_ligCode[ligCode] - length(sel_rm) < max_num_ligCode) {
      sel_rm <- sample(sel_rm, total_ligCode[ligCode] - max_num_ligCode)
    }
    
    ligs_data <- ligs_data[-sel_rm,]
    cat("\n  - Removed a total of", length(sel_rm), 
        "entries.")
  }
  t2 <- Sys.time()
  cat("\nDone in",round(t2-t1, 2), units(t2-t1), "!\n")
}

# filter the classes using the min occurrency by class
rm_ligs_by_labels <- c(1)
while(sum(rm_ligs_by_labels) > 0) {
  # labels_occ = colSums(ligs_data[,keep_labels])
  # names(labels_occ) <- vocab[as.numeric(keep_labels)+1]
  class_by_entry_ratio <- colSums(ligs_data[,keep_labels]>0)
  names(class_by_entry_ratio) <- vocab[as.numeric(keep_labels)+1]
  exclude_labels <- format(match(names(class_by_entry_ratio[class_by_entry_ratio < min_class_occ]), vocab)-1,
                           trim = TRUE)
  if (length(exclude_labels) == 0) { break() }
  cat("\n* Filtering the ligands that contains a class with less than the minimum occurrence threshold ")
  cat("\n    The ligands with the following labels will be removed:", paste(vocab[as.numeric(exclude_labels)+1], collapse = ","))
  rm_ligs_by_labels <- (rowSums(ligs_data[,exclude_labels]) > 0)
  cat("\n- Removed a total of", sum(rm_ligs_by_labels), 
      "ligands")
  ligs_data <- ligs_data[!rm_ligs_by_labels,]
  class_by_entry_ratio <- colSums(ligs_data[,keep_labels]>0)
  keep_labels <- keep_labels[class_by_entry_ratio > 0]
}
cat("\n\n* Entries by class occurrence ratio\n")
class_by_entry_ratio <- colSums(ligs_data[,keep_labels]>0)
names(class_by_entry_ratio) <- vocab[as.numeric(keep_labels)+1]
print(class_by_entry_ratio)
cat("\n* Total atoms by class occurrence ratio\n")
class_total_ratio <- colSums(ligs_data[,keep_labels])
names(class_total_ratio) <- vocab[as.numeric(keep_labels)+1]
print(class_total_ratio)

# undersample the classes that are above max_class
# fix the ligants that appear in the classes where the occurrence by entry > min and < max and then filter the remaning data
cat("\n* Filtering the classes with more than the maximum occurrence threshold, removing the entries at random ")
fix_labels <- format(match(names(class_by_entry_ratio[(class_by_entry_ratio >= min_class_occ & 
                                                         class_by_entry_ratio <= max_class_occ)]), vocab)-1, trim = TRUE)
ligs_data$keep_entry <- (rowSums(ligs_data[,fix_labels]) > 0)
# change the max class occurrence to the current max of the fixed labels
if (length(fix_labels) > 0) {
  max_class_occ <- max(class_by_entry_ratio[(class_by_entry_ratio >= min_class_occ & 
                                               class_by_entry_ratio <= max_class_occ)])
}
# check if the classes with >= max_class_occ are above this threshold in the set of keep_entry
labels_above_max <- format(match(names(class_by_entry_ratio[(class_by_entry_ratio > max_class_occ)]), vocab)-1, trim = TRUE)
# if any is below the threshold, increase this class
if (sum(ligs_data$keep_entry) > 0) {
  labels_above_max_occ <- colSums(ligs_data[ligs_data$keep_entry ,labels_above_max]>0)
} else {
  labels_above_max_occ <- rep(0, length(labels_above_max))
}
if (any(labels_above_max_occ < max_class_occ)) {
  # select the classes that are below the max occ
  labels_above_max_below_sel <- labels_above_max[labels_above_max_occ < max_class_occ]
  # order this labels increasing according to their occurrence in the keep set
  labels_above_max_below_sel <- labels_above_max_below_sel[order(class_by_entry_ratio[labels_above_max_below_sel])]
  # for each label in this order select the entries that should be included so its occurrence will hit the maximum
  for (label_increase in labels_above_max_below_sel) {
    # select the entries that are not included and that contain the labels selected above
    entries_sel <- (!ligs_data$keep_entry & (ligs_data[,label_increase] > 0))
    if (sum(ligs_data$keep_entry) > 0) {
      amount_increase <- max_class_occ-colSums(ligs_data[ligs_data$keep_entry ,label_increase]>0) #labels_above_max_occ[label_increase]
    } else {
      amount_increase <- max_class_occ
    }
    if (amount_increase <= 0) {
      next()
    }
    amount_can_increase <- sum(ligs_data[unlist(entries_sel), label_increase] > 0)
    if (amount_can_increase <= amount_increase) {
      # select all
      ligs_data$keep_entry[ligs_data[unlist(entries_sel), label_increase] > 0] <- TRUE
    } else {
      # select a sample from the available entries equals amount_increase
      # sample uniformally using the ligCode and bfactor_ratio, which were used to sort the data
      ligs_data$keep_entry[sample(which(entries_sel), amount_increase)] <- TRUE
    }
  }
}
cat("\n- Removed a total of", sum(!ligs_data$keep_entry), 
    "ligands")

cat("\n\n* Entries by class occurrence ratio\n")
class_by_entry_ratio <- colSums(ligs_data[ligs_data$keep_entry,keep_labels]>0)
names(class_by_entry_ratio) <- vocab[as.numeric(keep_labels)+1]
print(class_by_entry_ratio)
cat("\n* Total atoms by class occurrence ratio\n")
class_total_ratio <- colSums(ligs_data[ligs_data$keep_entry ,keep_labels])
names(class_total_ratio) <- vocab[as.numeric(keep_labels)+1]
print(class_total_ratio)

keep_labels <- keep_labels[class_by_entry_ratio > 0]
ligs_data <- ligs_data[ligs_data$keep_entry,]

cat("\n\n* Final number of ligand entries: ", nrow(ligs_data))

if (length(keep_labels) != num_classes) {
  # keep_labels <- paste0(vocab[as.numeric(keep_labels)+1], collapse = ",")
  keep_labels <- paste0(length(keep_labels), "classes")
} else {
  keep_labels <- "all"
}

min_class_occ <- min(class_by_entry_ratio)
max_class_occ <- max(class_by_entry_ratio)

write_csv(ligs_data, path=sub(x=ligs_data_path, pattern = ".csv", 
                              replacement = paste0("_undersampling_", keep_labels,
                                                   "_maxLigCode_",max_num_ligCode,
                                                   "_classOcc_",min_class_occ,
                                                   "_", max_class_occ,
                                                   ".csv")))
t2 <- Sys.time()
cat("\nFinish in",round(t2-t0, 2), units(t2-t0), "!\n\n")
