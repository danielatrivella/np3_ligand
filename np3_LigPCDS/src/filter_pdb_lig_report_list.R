suppressPackageStartupMessages(library(dplyr))
library(readr)
options(warning.length = 2000) # error msg length
ligand_report_path <- "PDB_lists/Ligands_PDB_entries_structure_factors_xray_protein.csv"
pdb_report_path <- "PDB_lists/PDB_entries_structure_factors_xray_protein.csv"

min_counts <- 1
min_resolution <- 1.5
max_resolution <- 2.2
np_ligands_filter <- TRUE
all_ligands <- FALSE # if false filter only Free ligands
date_filter <- "2008-02-01"

# read input
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 6) {
  stop("Wrong number of parameters. Six arguments must be supplied to filter the PDB report list and the ligand report list. ",
       "These reports must follow the table format exported by PDB in September 2025. \nParameters:\n",
       " 1. pdb_report: Path to the pre-filtered PDB structure report list with free ligands information - CSV table containing the PDB entries information, ",
       "mandatory columns (spaces are removed): PDBID, RefinementResolution, DepositionDate;\n",
       " 2. ligands_report: Path to the pre-filtered ligand report list with free ligands information - CSV table containing the ligands entries by ligand ID and PDB entries in which they appear, ",
       "with the following mandatory columns (spaces are removed): EntryID, ",
       "LigandID, LigandFormula, AsymID, AuthAsymID, freeLigand;\n",
       " 3. min_pdbids: Minimum number of PDB entries (IDs) in which a ligand must be present to be included in the resulting list of entries;\n",
       " 4. resolution_min: Minimum resolution of a PDB entry to be included in the resulting list;\n",
       " 5. resolution_max: Maximum resolution of a PDB entry to be included in the resulting list;\n",
       " 6. np_filter: TRUE or FALSE to apply the natural products filter and only ",
       "retain ligands that have the following organic atoms: C,H,O,N,P,S,I,Br,Cl,F,Se;\n",
       " 7. min_deposit_date: The minimum deposit date that a PDB entry must have to be included in the resulting list or an empty string to no deposit date filter. ",
       "All filtered entries must have been deposited after or in this date. ",
       "The informed date must follow the format yyyy-mm-dd, as in 2008-02-01, where yyyy is the year, mm is the month and dd is the day. Default to '2008-02-01'.\n",
       "8. all_ligands: TRUE or FALSE to indicate to keep all ligands or to filter only the Free ligands (non-covalent). Default to FALSE - filter the free ligand.",
       call. = FALSE)
} else {
  pdb_report_path <- file.path(args[[1]])
  if (!file.exists(pdb_report_path))
  {
    stop("The CSV file '", pdb_report_path,
         "' do not exists. Provide a valid path to where the PDB table is located.")
  }
  pdb_report_path <- normalizePath(pdb_report_path)
  
  ligand_report_path <- file.path(args[[2]])
  if (!file.exists(ligand_report_path))
  {
    stop("The CSV file '", ligand_report_path,
         "' do not exists. Provide a valid path to where the ligands table is located.")
  }
  ligand_report_path <- normalizePath(ligand_report_path)
  
  min_counts <- max(as.numeric(args[[3]]), 1)
  min_resolution <- as.numeric(args[[4]])
  max_resolution <- as.numeric(args[[5]])
  np_ligands_filter <- as.logical(args[[6]])
  
  if (length(args) > 6) {
    date_filter <- tryCatch(as.Date(args[[7]]), 
                            error = function(e) 
      stop("Failed to convert the deposit data filter to a Date object. ",
           "It must follow the format yyyy-mm-dd, as in 2008-02-01, where yyyy ",
           "is the year, mm is the month and dd is the day. Retry with correct ",
           "parameters. \nError msg: ", e, call. = FALSE))
    if (length(args) > 7) {
      all_ligands <- as.logical(args[[8]])
    }
  }
  
}


cat("** Filtering the ligands list - Free ligands, minimum count cutoff and NP atoms filter **\n")
# read ligands counts report from RCSB PDB and remove spaces and parentheses from header
ligands_report_list <- suppressMessages(read_csv(ligand_report_path,guess_max=50000))
cat("- Number of ligands in list = ", nrow(ligands_report_list),"\n\n")
names(ligands_report_list) <- gsub(" |\\(|\\)", "", names(ligands_report_list))

# filter non-covalent - free ligands
if (!all_ligands) {
  ligands_report_list <- ligands_report_list[ligands_report_list$freeLigand,]
  cat("- Number of filtered ligands after free ligands selection = ", nrow(ligands_report_list),"\n\n")
}

# filter the ligands by the minimum count
ligandID_count_value <- table(ligands_report_list$LigandID)
valid_ligandID <- names(ligandID_count_value)[ligandID_count_value >= min_counts]
ligands_report_list <- ligands_report_list[ligands_report_list$LigandID %in% valid_ligandID,]
cat("- Number of filtered ligands after minimum count cutoff filter = ", nrow(ligands_report_list),"\n\n")

# filter ligands that contain only atoms in CHONPS,I,Br,Cl,F,Se
if (np_ligands_filter) {
  np_atoms <- c('C','H','O','N','P','S','I','Br','Cl','F','Se')
  np_ligands_filter_test <- sapply(ligands_report_list$LigandFormula, 
    function(mf)
    {
      mf_atoms <-  strsplit(gsub("[0-9]*", "", perl = TRUE, mf), " ")[[1]]
      all(mf_atoms %in% np_atoms)
    })
  ligands_report_list <- ligands_report_list[np_ligands_filter_test,]
}
cat("- Number of filtered ligands after NP filter = ", nrow(ligands_report_list),"\n\n")

# read the available pdb entries and remove spaces and parentheses from header
pdb_report_list <- suppressMessages(read_csv(pdb_report_path))
names(pdb_report_list) <- gsub(" |\\(|\\)|\\.", "", names(pdb_report_list))
pdb_report_list$PDBID <- toupper(pdb_report_list$PDBID)

cat("- Number of PDB entries in list = ", nrow(pdb_report_list),"\n\n")

cat("** Filtering the PDB list - by the entries kept in the Ligands list **\n")
#  now, filter the kept pdb ids from the ligands report list
pdb_report_list <- pdb_report_list[pdb_report_list$PDBID %in% ligands_report_list$EntryID,]
cat("- Number of filtered PDB entries after ligands list filter = ", nrow(pdb_report_list),"\n\n")

cat("** Filtering the PDB list - by resolution range and deposit date **\n")

# filter the pdb entries by the provided resolution range
pdb_report_list <- pdb_report_list[pdb_report_list$RefinementResolution >= min_resolution &
                           pdb_report_list$RefinementResolution <= max_resolution,]

cat("- Number of filtered PDB entries after resolution limit = ", 
    nrow(pdb_report_list),"\n\n")
# filter by deposit date
if (class(date_filter) == "Date") {
  pdb_report_list <- pdb_report_list[pdb_report_list$DepositionDate >= date_filter,]
} else {
  # if no date was informed, then retrieve the minimum date in the list
  date_filter <- min(pdb_report_list$DepositionDate)
}
date_filter <- as.character(date_filter)

cat("- Number of filtered PDB entries after deposit date (", date_filter,
    ") limit = ", nrow(pdb_report_list),"\n\n",sep="")

cat("** Filtering the ligands that are present in the filtered PDB entries by resolution and deposity date **\n")

ligands_report_list <- ligands_report_list[ligands_report_list$EntryID %in% pdb_report_list$PDBID,]
cat("- Number of filtered ligands after resolution and deposit date filter = ", nrow(ligands_report_list),"\n\n")

# filter the ligands by the minimum count again
ligandID_count_value <- table(ligands_report_list$LigandID)
valid_ligandID <- names(ligandID_count_value)[ligandID_count_value >= min_counts]
ligands_report_list <- ligands_report_list[ligands_report_list$LigandID %in% valid_ligandID,]
cat("- Number of filtered ligands after second minimum count cutoff filter = ", nrow(ligands_report_list),"\n\n")
cat("** Filtering the PDB list again - by the entries kept in the Ligands list\n")
#  now, filter the kept pdb ids from the ligands report list again
pdb_report_list <- pdb_report_list[pdb_report_list$PDBID %in% ligands_report_list$EntryID,]
cat("- Number of filtered PDB entries after ligands list filter = ", nrow(pdb_report_list),"\n\n")

cat("** Counting the number of ligands in Asym and AuthAsym chains and computing this count by PDB ID **\n")
# count in the ligands report list the number of occurrences of each 
## ligand in each PDB ID by asymID - number of chains given by PDB (usually one different for each ligand appearence, instead of the author chain ID)
# then count in the pdb report list, the total number of ligands for each PDB ID
ligands_report_list$numTotalCount <- sapply(ligands_report_list$AsymID, function(x) {
  if (!is.na(x))
    length(strsplit(x,", ")[[1]])
  else
    0
  }, USE.NAMES = FALSE)


for (i in seq_len(nrow(pdb_report_list))) {
  pdbid <- pdb_report_list[i,"PDBID"][[1]]
  ligs_pdbid <- ligands_report_list[ligands_report_list$EntryID == pdbid,]
  num_distinctLigands <- nrow(ligs_pdbid)
  num_totalLigands <- sum(ligs_pdbid$numTotalCount)
  num_distinctFreeLigands <- nrow(ligs_pdbid[ligs_pdbid$freeLigand,])
  if (num_distinctFreeLigands > 0)
    num_totalFreeLigands <- sum(ligs_pdbid[ligs_pdbid$freeLigand,"numTotalCount"])
  else
    num_totalFreeLigands <- 0
  pdb_report_list[i,"NumberofDistinctLigands"] <- num_distinctLigands
  pdb_report_list[i,"TotalNumberofLigands"] <- num_totalLigands
  pdb_report_list[i,"NumberofDistinctFreeLigands"] <- num_distinctFreeLigands
  pdb_report_list[i,"TotalNumberofFreeLigands"] <- num_totalFreeLigands
}

cat("** Reports list summary **\n\n")

cat("- Final number of PDB entries = ", nrow(pdb_report_list))
cat("\n- Final Number of ligands (",ifelse(all_ligands, "all", "free"),") = ", 
    nrow(ligands_report_list),"\n\n")
cat("- Final Number of *unique* ligands (",ifelse(all_ligands, "all", "free"),") = ", 
    length(unique(ligands_report_list$LigandID)),"\n\n")

cat("** PDB list summary **\n\n")

cat("- Sum of the total number of ligands in Asym and AuthAsym units that appear in the list of filtered PDB IDs= ", 
    sum(pdb_report_list$TotalNumberofLigands),
    "\n    - This is the minimum number of ligand entries, each ligand can ",
    "appear more than once in a unit of a PDB entry\n")
cat("Sum of the total number of distinct (unique) ligands that appear in the list of filtered PDB IDs= ", 
    sum(pdb_report_list$NumberofDistinctLigands),
    "\n  - This is the sum of distinct ligands by PDB entry\n")
cat("\nSummary of the Total Number of Ligands by PDB entry = \n")
print(c(summary(pdb_report_list$TotalNumberofLigands), Sd. = sd(pdb_report_list$TotalNumberofLigands)))

write_csv(pdb_report_list, file = paste("PDB",min_resolution,max_resolution,
                                    ifelse(np_ligands_filter, 
                                           "NP",
                                           #paste0(np_atoms, collapse = ""), 
                                           "all"), "atoms",
                                    ifelse(all_ligands, "all", "free"), "ligands", 
                                    min_counts, "counts", 
                                    ifelse(date_filter=="", 
                                           ".csv",
                                           paste0(date_filter,"_depDate.csv")), 
                                    sep = "_"))
write_csv(ligands_report_list, file = paste("ligands", ifelse(all_ligands, "all", "free"),
                                       "PDB", min_resolution,max_resolution,
                                       ifelse(np_ligands_filter, 
                                              "NP",
                                              #paste0(np_atoms, collapse = ""), 
                                              "all"), "atoms",
                                       min_counts, "counts", 
                                       ifelse(date_filter=="", 
                                              ".csv",
                                              paste0(date_filter,"_depDate.csv")), 
                                       sep = "_"))
