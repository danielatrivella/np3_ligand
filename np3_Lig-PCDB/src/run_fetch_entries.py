from fetch_pdb_mtz_entries import fetch_pdb_mtz
from fetch_sdf_ligands_pdb_res import fetch_sdf_ligands

if __name__ == "__main__":
    import sys

    i_start = 0
    n = 0
    if len(sys.argv) >= 3:
        db_path = sys.argv[1]
        pdb_ligs_file = sys.argv[2]
        if len(sys.argv) > 3:
            i_start = int(sys.argv[3])
        if len(sys.argv) > 4:
            n = int(sys.argv[4])
    else:
        sys.exit("Wrong number of arguments. At least two arguments must be supplied in order to retrieve the entries "
                 ".pdb, .mtz and ligands .sdf files and to create the .pdb's of each ligands and the entries "
                 "pdb's without each ligand: \n"
                 "  1. The path to the data folder where the data will be stored. Four folders will be created: "
                 "'pdb' to store the .pdb files; 'coefficients' to store the .mtz files; 'ligands' to store "
                 "the .sdf and the .pdb files of the ligands; and 'pdb_no_lig' to store the .pdb files without each "
                 "ligand coordinates;\n"
                 "  2. The path to the CSV file containing the PDB IDs of the entries and the "
                 "ligands of each entry to be retrieved. Mandatory columns = 'PDBID' and 'ligandsID'. "
                 "The column 'ligandsID' must have the names separated by a space of all the desired ligands present "
                 "in the structure with code equals 'PDBID';\n"
                 "  3. (optional) The number of the row of the entries CSV file where the script should start. "
                 "Skip to the given row or, if missing, start from the beginning;\n"
                 "  4. (optional) The number of the row of the ligands CSV file where the script should stop. "
                 "Stop in the given row or, if missing, stop in the last row.")

    print("\n****************************************")
    print("*** Running the PDB Entries Download ***")
    print("****************************************\n")
    fetch_pdb_mtz(db_path, pdb_ligs_file, n, i_start)

    print("\n\n****************************************")
    print("*** Running the Ligands sdf Download ***")
    print("****************************************\n")
    fetch_sdf_ligands(db_path, pdb_ligs_file, n, i_start)

    print("\nDONE!")

