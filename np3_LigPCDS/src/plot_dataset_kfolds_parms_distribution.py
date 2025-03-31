import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_kfold_dist(ligs_path,vocab_path,mapping_path):
    ligs_path = Path(ligs_path)
    ligs = pd.read_csv(ligs_path)
    vocab = [line.rstrip('\n') for line in open(vocab_path)]

    # apply mapping if the table was provided
    if mapping_path:
        mapping_path = Path(mapping_path)
        mapping = pd.read_csv(mapping_path).sort_values('source')
        mapping = mapping.loc[0:len(vocab) - 1, ]  # remove background
        new_vocab = mapping.target.unique()
        new_vocab.sort()
        new_vocab = pd.DataFrame(new_vocab)
        vocab0_cols = [ligs.columns.get_loc(str(x)) for x in list(range(len(vocab)))]
        dist_old_vocab = ligs.iloc[:,vocab0_cols]
        ligs = ligs.drop(ligs.columns[vocab0_cols], axis=1)
        # aggregate the distribution of the new vocab based on the mapping and save to the ligands entries file
        for new_label_idx in new_vocab[0]:
            ligs[str(new_label_idx)] = dist_old_vocab[
                mapping.source[mapping.target == new_label_idx].values.astype(str)].sum(1)
        # replace old vocab with new
        vocab = mapping[~mapping[['mapping', 'target']].duplicated()].sort_values('target').mapping.reset_index(drop=True).values.tolist()
        del dist_old_vocab
        # rename skipping background (index 0)
        ligs.rename(columns={str(i): x for i, x in enumerate(vocab,1)}, inplace=True)
    else:
        ligs.rename(columns={str(i):x for i,x in enumerate(vocab)}, inplace=True)

    # transform min occ to percentage
    ligs['min_occ_percentage'] = round(ligs.min_occupancy * 100)
    # select the variables to be plotted
    var_cols_selected = ["bfactor_ratio", "Resolution", "min_occ_percentage", "point_cloud_size_qRankMask"] + vocab  # removed  "ligCode", "entry" que sao chars

    path_distr = (ligs_path.parent / (ligs_path.as_posix().replace(".csv", "") +
                  ("_"+mapping_path.name.replace(".csv", "") if mapping_path else "") + "_kfold_distributions"))
    if not path_distr.exists():
        path_distr.mkdir()
    # plot each var boxplot in terms of kfolds and test_val
    for x in var_cols_selected:
        print("Plotting the variable", x, "boxplot distribution")
        sns.set_theme(style="darkgrid")
        # sns.displot(
        #     ligs, x=x, col="test_val", row="kfolds",  shrink=.9, discrete=True, multiple="dodge",
        #     binwidth=2, height=3, facet_kws=dict(margin_titles=True),
        # )
        sns.boxplot(data=ligs, x=x, y="kfolds", hue="test_val", orient="h", whis=2, showfliers = False)
        plt.savefig(path_distr / ('distribution_numeric_var_'+x+'.png'))
        plt.close()
        # plt.show()


if __name__ == "__main__":
    mapping_path = None
    if len(sys.argv) >= 3:
        ligs_path = sys.argv[1]
        vocab_path = sys.argv[2]
        if len(sys.argv) >= 4:
            mapping_path = sys.argv[3]
    else:
        sys.exit("Wrong number of arguments. Two arguments must be supplied in order to plot the kfolds distributions "
                 "accross different variables: \n"
                 "  1. The path to the CSV file containing the valid ligands list and their classes frequency. "
                 "This file is expected to be the output of the encode_ligs_xyz script. It should be located in the "
                 "'xyz_<ligand csv name>' folder inside the 'ligands' folder, named with the suffix '_class_freq.csv' or "
                 "other file with the vocab classes indices as column names\n"
                 "  2. The path to the text file containing the vocabulary used to label the ligands. "
                 "It must contain one class per line. \n"
                 "  3. (optional) The path to the CSV file containing a mapping between the vocabulary classes and the simplified"
                 "classes. Mandatory columns: source (old vocab indexes), target (new vocab indexes), "
                 "mapping (new vocab classes names). The last row must be the mapping for the background class, which is ignored. \n"
                 )
    plot_kfold_dist(ligs_path,vocab_path,mapping_path)