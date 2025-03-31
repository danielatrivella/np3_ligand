import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dist(data, col, output_path):
    sns.set_theme(style="darkgrid")
    sns.displot(data, x=col, height=5)
    # plt.show()
    plt.savefig(output_path / ('distribution_' + col + '.png'))

def compute_ligs_sym_stats(ligs_data_path, pc_path):
    pc_path = Path(pc_path)
    ligs_data_path = Path(ligs_data_path)
    ligs_data = pd.read_csv(ligs_data_path, usecols=['ligID', 'entry'])

    for i in range(ligs_data.shape[0]):
        lig_grid_sym_mates_file = pc_path / ligs_data.loc[i,'entry'] / (ligs_data.loc[i,'ligID']+
                                                                    '_grid_point_cloud_fofc_symmetric_mates.csv')
        if not lig_grid_sym_mates_file.exists():
            continue
        # read the ligand grid symmetric mates rho
        lig_grid_sym_mates = pd.read_csv(lig_grid_sym_mates_file)
        # if any symmetric mate is present compute the rho diff describe mean
        if lig_grid_sym_mates.columns.str.startswith("rho_").any():
            sym_mates_describe = lig_grid_sym_mates.loc[:, lig_grid_sym_mates.columns.str.startswith("rho_")].subtract(
                lig_grid_sym_mates["rho"],axis="index").describe()
            # set the mean to the ligands data
            ligs_data.loc[i, sym_mates_describe.index[1:]] = [sym_mates_describe.loc['mean',].mean(),
                                                              sym_mates_describe.loc['std',].mean(),
                                                              sym_mates_describe.loc['min',].min(),
                                                              sym_mates_describe.loc['25%',].mean(),
                                                              sym_mates_describe.loc['50%',].mean(),
                                                              sym_mates_describe.loc['75%',].mean(),
                                                              sym_mates_describe.loc['max',].max()]


    print("Symmetric mate rho mean = ", ligs_data['mean'].mean())
    print("Symmetric mate rho min = ", ligs_data['min'].min())
    print("Symmetric mate rho max = ", ligs_data['max'].max())

    output_path = ligs_data_path.parent / ligs_data_path.name.replace(".csv", "_sym_mates_dist")
    output_path.mkdir(exist_ok=True)

    ligs_data.to_csv(output_path / "ligs_sym_mates_desc.csv")

    for x in sym_mates_describe.index[1:]:
        plot_dist(ligs_data, x, output_path)



if __name__ == "__main__":
    if len(sys.argv) >= 3:
        ligands_file_path = sys.argv[1]
        pc_path = sys.argv[2]
    else:
        sys.exit("Wrong number of arguments. Two arguments are necessary to plot the ligands symmetric mates rho difference distribution: \n"
                 "  1. The path to the CSV file containing the ligands to be used in the distribution plot.\n"
                 "  2. The path to the folder where the point clouds are located. \n"
                 )
    compute_ligs_sym_stats(ligands_file_path, pc_path)
