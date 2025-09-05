from pathlib import Path
import pandas as pd
import sys


def concatenate_tables(dir_path, output_report_name="rcsb_pdb_custom_report_all.csv", files_pattern="rcsb_pdb_custom_report_*.csv", n_skip_rows=1):
    dir_path = Path(dir_path)

    report_files_path =sorted(list(dir_path.glob(files_pattern)))

    report_tables = pd.concat([pd.read_csv(report_file, skiprows=n_skip_rows,skip_blank_lines=True)
                     for report_file in report_files_path])

    report_tables.to_csv(dir_path/output_report_name, index=False)


if __name__ == "__main__":
    output_report_name = "rcsb_pdb_custom_report_all.csv"
    files_pattern = "rcsb_pdb_custom_report_*.csv"
    n_skip_rows = 1
    if len(sys.argv) >= 2:
        dir_path = sys.argv[1]
        if len(sys.argv) >= 3:
            output_report_name = sys.argv[2]
        if len(sys.argv) >= 4:
            files_pattern = sys.argv[3]
        if len(sys.argv) >= 5:
            n_skip_rows = sys.argv[4]
    else:
        sys.exit("Wrong number of arguments. One argument is necessary to concatenate all csv reports inside a given directory and store the result inside it: \n"
                 "  1. dir_path: The path to the diretory containing the report tables and where the output concatenated report table will be stored;\n"
                 "  2. output_report_name: default to rcsb_pdb_custom_report_all.csv - structure report output name;\n"
                 "  3. files_pattern: default to rcsb_pdb_custom_report_*.csv - structure report file pattern;\n"
                 "  4. n_skip_rows: number of initial rows to skip from the reports, default to 1.\n"
                 )
    concatenate_tables(dir_path, output_report_name, files_pattern, n_skip_rows)