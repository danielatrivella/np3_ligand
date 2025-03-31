suppressPackageStartupMessages(library(dplyr))
library(readr)
options(warning.length = 2000) # error msg length
ligand_path <- "PDB_lists/Ligands_PDB_entries_structure_factors_xray_protein.csv"
pdb_path <- "PDB_lists/PDB_entries_structure_factors_xray_protein.csv"

min_counts <- 10
min_counts <- max(min_counts, 1)
min_resolution <- 1.0
max_resolution <- 1.5
np_ligands_filter <- TRUE
all_ligands <- FALSE # if false filter only Free ligands
date_filter <- ""

# read input
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 6) {
  stop("Wrong number of parameters. Six arguments must be supplied to filter the PDB report list and the ligand report list. ",
       "These reports must follow the table format exported by PDB in july 2019. \nParameters:\n",
       " 1. pdb_report: Path to the PDB report list - CSV table containing the PDB entries information, ",
       "mandatory columns (spaces are removed): PDBID, Resolution;\n",
       " 2. ligand_report: Path to the ligand report list - CSV table containing the ligands entries and their counts by code and by PDB entries in which they appear, ",
       "with the following mandatory columns (spaces are removed): LigandID, ",
       "LigandFormula, PDBIDsTotal, PDBIDsFreeLigand, InstancePDBIDsasFreeLigand, InstancePDBIDsAll;\n",
       " * Only the Free ligands are used (InstancePDBIDsasFreeLigand), the polymeric ligands are ignored;\n",
       " 3. min_pdbids: Minimum number of PDB entries (IDs) in which a ligand must be present to be included in the resulting list of entries;\n",
       " 4. resolution_min: Minimum resolution of a PDB entry to be included in the resulting list;\n",
       " 5. resolution_max: Maximum resolution of a PDB entry to be included in the resulting list;\n",
       " 6. np_filter: TRUE or FALSE to apply the natural products filter and only ",
       "retain ligands that have the following organic atoms: C,H,O,N,P,S,I,Br,Cl,F,Se;\n",
       " 7. min_deposit_date: The minimum deposit date that a PDB entry must have to be included in the resulting list (default to no deposit date filter). ",
       "All filtered entries must have been deposited after or in this date. ",
       "The informed date must follow the format yyyy-mm-dd, as in 2008-01-01, where yyyy is the year, mm is the month and dd is the day.\n",
       call. = FALSE)
} else {
  pdb_path <- file.path(args[[1]])
  if (!file.exists(pdb_path))
  {
    stop("The CSV file '", pdb_path,
         "' do not exists. Provide a valid path to where the PDB table is located.")
  }
  pdb_path <- normalizePath(pdb_path)
  
  ligand_path <- file.path(args[[2]])
  if (!file.exists(ligand_path))
  {
    stop("The CSV file '", ligand_path,
         "' do not exists. Provide a valid path to where the ligands table is located.")
  }
  ligand_path <- normalizePath(ligand_path)
  
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
  }
  # all_ligands <- as.logical(args[[7]])
}


cat("** Filtering the ligands list - minimum count cutoff and NP atoms filter\n")
# read ligands counts report from RCSB PDB and remove spaces and parentheses from header
ligands_counts <- suppressMessages(read_csv(ligand_path))
cat("Number of ligands in list = ", nrow(ligands_counts),"\n\n")
names(ligands_counts) <- gsub(" |\\(|\\)", "", names(ligands_counts))

# select the PDB IDs to be used; set them to upper case; and filter the ligands by the minimum count
if (all_ligands) {
  ligs_selection <- 'InstancePDBIDsAll'
  ligands_counts <- ligands_counts[ligands_counts$PDBIDsTotal >= min_counts,]
} else {
  ligs_selection <- 'InstancePDBIDsasFreeLigand'
  ligands_counts <- ligands_counts[ligands_counts$PDBIDsFreeLigand >= min_counts,]
}
ligands_counts[[ligs_selection]] <- toupper(ligands_counts[[ligs_selection]])
cat("Number of filtered ligands after minimum count cutoff = ", nrow(ligands_counts),"\n\n")

# filter ligands that contain only atoms in CHONPS,I,Br,Cl,F,Se
if (np_ligands_filter) {
  np_atoms <- c('C','H','O','N','P','S','I','Br','Cl','F','Se')
  np_ligands_filter <- sapply(ligands_counts$LigandFormula, 
    function(mf)
    {
      mf_atoms <-  strsplit(gsub("[0-9]*", "", perl = TRUE, mf), " ")[[1]]
      all(mf_atoms %in% np_atoms)
    })
  ligands_counts <- ligands_counts[np_ligands_filter,]
  np_ligands_filter <- TRUE
}
cat("Number of filtered ligands after NP filter = ", nrow(ligands_counts),"\n\n")

# read the available pdb entries and remove spaces and parentheses from header
pdb_entries <- suppressMessages(read_csv(pdb_path))
names(pdb_entries) <- gsub(" |\\(|\\)|\\.", "", names(pdb_entries))
pdb_entries$PDBID <- toupper(pdb_entries$PDBID)

cat("** Filtering the PDB list - by resolution range and deposit date\n")

cat("Number of PDB entries in list = ", nrow(pdb_entries),"\n\n")

# filter the pdb entries by the provided resolution range
pdb_entries <- pdb_entries[pdb_entries$Resolution >= min_resolution &
                           pdb_entries$Resolution <= max_resolution,]

cat("Number of filtered PDB entries after resolution limit = ", nrow(pdb_entries),"\n\n")
# filter by deposit date
if (class(date_filter) == "Date") {
  pdb_entries <- pdb_entries[pdb_entries$DepDate >= date_filter,]
} else {
  date_filter <- min(pdb_entries$DepDate)
}
date_filter <- as.character(date_filter)

cat("Number of filtered PDB entries after deposit date limit = ", nrow(pdb_entries),"\n\n")

cat("** Filtering the ligands that are present in the filtered PDB entries,",
    "counting the remaining number of PDB entries where each ligand appears and",
    "applying the minimum count cutoff\n")
 
# for each ligand filter the pdb codes that are in the given resolution range
# and return their ids and count
# if the count is greater then the min count, attribute to each pdb entry
# the ligand appearence and compute its count
pdb_entries$ligandsID <- ""
pdb_entries$ligands_count <- 0
ligands_counts[,c("PDB_IDs", "count")] <- bind_rows(
  lapply(seq_len(nrow(ligands_counts)), 
         function(i, valid_codes = pdb_entries$PDBID, min_count = min_counts) {
           x <- ligands_counts[[ligs_selection]][i]
           codes <- strsplit(x, split = ",")[[1]]
           codes_resolution_idx <- match(codes, valid_codes)
           codes_resolution <- which(!is.na(codes_resolution_idx))
           n <- length(codes_resolution)
           # codes_resolution <- which(codes %in% valid_codes)
           
           if (n > 0) {
             # if the ligand is going to be included in the list count its occurence 
             # in the corresponding pdb entries where it appears
             if (n >= min_count) {
               codes_resolution_idx <- codes_resolution_idx[codes_resolution]
               pdb_entries$ligands_count[codes_resolution_idx] <<- 
                 pdb_entries$ligands_count[codes_resolution_idx] + 1
               pdb_entries$ligandsID[codes_resolution_idx] <<- 
                 paste(pdb_entries$ligandsID[codes_resolution_idx], 
                       ligands_counts$LigandID[[i]])
             } 
             
             list(PDB_IDs = paste(codes[codes_resolution], collapse = ","), 
                  count = n)
           } else {
             list(PDB_IDs = NA, count = 0)
           }
         }))

# filter ligands with at least one count and greater than min_counts
ligands_counts <- ligands_counts[ligands_counts$count >= min_counts,]
# filter PDB entries with at least one ligand count and trim white space
pdb_entries <- pdb_entries[pdb_entries$ligands_count > 0,]
pdb_entries$ligandsID <- trimws(pdb_entries$ligandsID)

cat("Final number of PDB entries = ", nrow(pdb_entries))
cat("\nFinal Number of ligands = ", nrow(ligands_counts),"\n\n")
cat("** PDB list summary\n\n")

cat("Sum of PDB entries in which the ligands appear = ", sum(ligands_counts$count),
    "\n  - This is the minimum number of ligands entries, each ligand can ",
    "appear more than once in a single PDB entry\n")
cat("\nSummary of the Count of PDB entries in which the ligands appear= \n")
print(c(summary(ligands_counts$count), Sd. = sd(ligands_counts$count)))

write.csv(pdb_entries, file = paste("PDB",min_resolution,max_resolution,
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
write.csv(ligands_counts, file = paste("ligands", ifelse(all_ligands, "all", "free"),
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
