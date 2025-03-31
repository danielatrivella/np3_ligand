import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

def ligs_class_freq_stats(ligand_classes, vocab0_cols, vocab, output_path, stats_file_prefix="ligands_atoms"):
    # create a data frame to store the classes statistics
    # sum the total frequency of each class over all ligands and their counts
    ligand_classes_freq_stats = pd.DataFrame.from_records([list(ligand_classes.columns[vocab0_cols]),
                                                           list(ligand_classes.iloc[:, vocab0_cols].sum()),
                                                           list(ligand_classes.iloc[:, vocab0_cols].std())]).transpose()
    ligand_classes_freq_stats.columns = ['vocab_class', 'total_count', 'total_std']
    # get the number of unique ligands entries by class
    ligand_classes = ligand_classes.iloc[:, vocab0_cols]
    ligand_classes[ligand_classes > 1] = 1
    ligand_classes_freq_stats["unique_ligands"] = list(ligand_classes.sum(0))
    # add vocab smiles
    ligand_classes_freq_stats["vocab_sp"] = vocab
    # save stats file
    ligand_classes_freq_stats.to_csv((output_path / str("classes_statistics_by_labeled_"+stats_file_prefix+".csv")),
                                     columns=['vocab_class', 'total_count', 'total_std', 'unique_ligands', 'vocab_sp'])
    # sort by unique ligands
    ligand_classes_freq_stats = ligand_classes_freq_stats.sort_values('unique_ligands')
    return ligand_classes_freq_stats

# used to change some of the plots settings for larger plots: increase the image size and rotate the x axis labels
# this is set to True when no mapping is informed
global large_plot
large_plot = False

def save_class_stats_plot(ligand_classes_freq_stats, ligands_classes_file_path, output_path, title_prefix="ligands_atoms",
                          legend_plot = ['Number of unique and labeled ligands code','Number of labeled atoms']):
    # plot a stacked barplot
    # Set general plot properties
    sns.set(style="whitegrid")
    sns.set_context({"figure.figsize": (12, 8)})
    # set for larger plots
    if large_plot:
        sns.set_context({"figure.figsize": (16, 10)})
    # def funEst(x):
    #     return ligand_classes_freq_stats[ligand_classes_freq_stats.total_count == x.iloc[0]]['total_std'].iloc[0]
    # Plot 1 - background - "total" (top) series
    sns.barplot(x=ligand_classes_freq_stats.vocab_sp, y=ligand_classes_freq_stats.total_count, color="#F15E5E")  # \
    #            estimator=lambda x: funEst(x), ci=None)
    # Plot 2 - overlay - "unique" (bottom) series
    bottom_plot = sns.barplot(x=ligand_classes_freq_stats.vocab_sp, y=ligand_classes_freq_stats.unique_ligands,
                              color="#595757")
    topbar = plt.Rectangle((0, 0), 1, 1, fc="#F15E5E", edgecolor='none')
    bottombar = plt.Rectangle((0, 0), 1, 1, fc="#595757", edgecolor='none')
    l = plt.legend([bottombar, topbar], legend_plot, loc='upper left', ncol=1,
                   prop={'size': 16})#, bbox_to_anchor=(0, 1.1)) # add to put the legend above the plot - larger plots
    l.draw_frame(False)
    #plt.suptitle(title_prefix+" SP Classes Statistics", fontsize=30)
    #plt.title(ligands_classes_file_path.name.replace(".csv", "").replace("_", " "), fontsize=20)
    # Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_yscale("log")
    bottom_plot.set_ylabel("Count")
    bottom_plot.set_xlabel("Classes")
    # Set fonts to consistent 20pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(20)

    # use to rotate the legend and to use a smaller font size in the xticks - larger plots
    if large_plot:
        for item in bottom_plot.get_xticklabels():
            item.set_fontsize(14)
        plt.xticks(rotation= 45)

    plt.savefig(output_path / ('classes_distribution_by_labeled_'+title_prefix+'.png'))
    plt.show()

def compute_plot_SP_classes_stats(ligands_classes_file_path, vocabulary_path, count_cut_off, mapping_path=None):
    ligands_classes_file_path = Path(ligands_classes_file_path)

    # read input
    ligand_entries = pd.read_csv(ligands_classes_file_path)
    vocab = pd.read_csv(vocabulary_path, header=None)

    # apply mapping if the table was provided
    if mapping_path:
        mapping_path = Path(mapping_path)
        output_path = ligands_classes_file_path.parent / str(ligands_classes_file_path.name.replace(".csv", "_") +
                                                             mapping_path.name.replace(".csv", ""))
        output_path.mkdir(exist_ok=True)
        mapping = pd.read_csv(mapping_path).sort_values('source')
        mapping = mapping.loc[0:vocab.shape[0] - 1, ]  # remove background
        new_vocab = mapping.target.unique()
        new_vocab.sort()
        new_vocab = pd.DataFrame(new_vocab)
        vocab0_cols = [ligand_entries.columns.get_loc(str(x)) for x in list(range(len(vocab)))]
        dist_old_vocab = ligand_entries.iloc[:,vocab0_cols]
        ligand_entries = ligand_entries.drop(ligand_entries.columns[vocab0_cols], axis=1)
        # aggregate the distribution of the new vocab based on the mapping and save to the ligands entries file
        for new_label_idx in new_vocab[0]:
            ligand_entries[str(new_label_idx)] = dist_old_vocab[
                mapping.source[mapping.target == new_label_idx].values.astype(str)].sum(1)
        # replace old vocab with new
        vocab = pd.DataFrame({0: mapping[~mapping[['mapping', 'target']].duplicated()].sort_values('target').mapping.reset_index(drop=True)})
        del dist_old_vocab
    else:
        global large_plot
        large_plot = True 
        output_path = ligands_classes_file_path.parent / ligands_classes_file_path.name.replace(".csv", "")
        output_path.mkdir(exist_ok=True)

    # sum the class frequency of each ligand code
    ligand_classes = ligand_entries.groupby('ligCode').agg('sum')
    # if mapping is present use the target column to get new vocab column names
    if mapping_path:
        vocab0_cols = [ligand_classes.columns.get_loc(str(x)) for x in new_vocab[0].values]
    else:
        vocab0_cols = [ligand_classes.columns.get_loc(str(x)) for x in list(range(len(vocab)))]
    ligand_classes_freq_stats = ligs_class_freq_stats(ligand_classes, vocab0_cols, vocab,
                                                      output_path, stats_file_prefix="ligands_atoms")
    save_class_stats_plot(ligand_classes_freq_stats, ligands_classes_file_path, output_path)
    # impact in the distribution when removing the ligands where less frequent classes appears
    # only select ligands that do not appear in a SP class with total count < count_cut_off
    if ((ligand_classes_freq_stats.total_count < count_cut_off) & (ligand_classes_freq_stats.total_count > 0)).any():
        ligand_classes = ligand_classes[(ligand_classes.loc[:,
                                         (ligand_classes_freq_stats.vocab_class[(ligand_classes_freq_stats.total_count < count_cut_off) &
                                                                                (ligand_classes_freq_stats.total_count > 0)])].sum(1) <= 0)]
        ligand_classes_freq_stats_greater_cutoff = ligs_class_freq_stats(ligand_classes,
                                                                         vocab0_cols, vocab, output_path, stats_file_prefix="ligands_atoms>"+str(count_cut_off))
        save_class_stats_plot(ligand_classes_freq_stats_greater_cutoff, ligands_classes_file_path, output_path, "ligands_atoms>"+str(count_cut_off))

    # sum class frequency of each ligand entry
    ligand_classes_entry = ligand_entries.groupby('ligCode').agg(np.count_nonzero)
    # if mapping is present use the target column to get new vocab column names
    if mapping_path:
        vocab0_cols = [ligand_classes_entry.columns.get_loc(str(x)) for x in new_vocab[0].values]
    else:
        vocab0_cols = [ligand_classes_entry.columns.get_loc(str(x)) for x in list(range(len(vocab)))]
    ligand_classes_freq_stats = ligs_class_freq_stats(ligand_classes_entry, vocab0_cols, vocab,
                                                      output_path, stats_file_prefix="ligands_entries")
    save_class_stats_plot(ligand_classes_freq_stats, ligands_classes_file_path, output_path, "ligands_entries",
                          ['Number of unique and labeled ligands code','Number of labeled ligands entries'])
                          #['Tipo de ligantes únicos rotulados','Ligantes rotulados'])

    if ((ligand_classes_freq_stats.total_count < count_cut_off) & (ligand_classes_freq_stats.total_count > 0)).any():
        ligand_classes_entry = ligand_classes_entry[(ligand_classes_entry.loc[:, (ligand_classes_freq_stats.vocab_class[
            (ligand_classes_freq_stats.total_count < count_cut_off) & (ligand_classes_freq_stats.total_count > 0)])].sum(1) <= 0)]
        ligand_classes_freq_stats_greater_cutoff = ligs_class_freq_stats(ligand_classes_entry,
                                                                         vocab0_cols, vocab, output_path,
                                                                         stats_file_prefix="ligands_entries>" + str(count_cut_off))
        save_class_stats_plot(ligand_classes_freq_stats_greater_cutoff, ligands_classes_file_path, output_path,
                              "ligands_entries>" + str(count_cut_off),
                              ['Number of unique and labeled ligands code','Number of labeled ligands entries'])#['Tipo de ligantes únicos rotulados','Ligantes rotulados'])


# ligCode_10k = ligand_classes.index.to_list()

if __name__ == "__main__":
    mapping_path = None
    if len(sys.argv) >= 4:
        ligands_classes_file_path = sys.argv[1]
        vocabulary_path = sys.argv[2]
        count_cut_off = float(sys.argv[3])
        if len(sys.argv) >= 5:
            mapping_path = sys.argv[4]
    else:
        sys.exit("Wrong number of arguments. Four parameters must be supplied in order to plot the classes distribution "
                 "and statistics of a vocabulary in a list of ligands entries: \n"
                 "  1. list_ligands_path: The path to the CSV file containing a list of ligands and their classes "
                 "frequency by column. This file is expected to be the output of the 'run_vocabulary_encode_ligands.py' "
                 "script. It should be located in the 'ligands/xyz_<ligand_list_path.name>' folder, "
                 "named with the suffix '_class_freq.csv' or other table with the indices of the vocabulary classes as column names;\n"
                 "  2. vocab_path: The path to the text file containing the vocabulary classes used to label the list of ligands. "
                 "It must contain one class by row; \n"
                 "  3. min_entry_occurrence: The minimum number of ligands entries occurrences that the classes must "
                 "have to be used in the distributions (do not use the ligands entries that have a label from a classe "
                 "with an occurrence by entry smaller than this cutoff);\n"
                 "  4. class_mapping_path: (optional) The path to the CSV file containing a mapping between the "
                 "vocabulary classes and the simplified classes. Mandatory columns: source, target. "
                 "The last row must be the mapping for the background class, which is not used in these distributions. \n"
                 )
    compute_plot_SP_classes_stats(ligands_classes_file_path, vocabulary_path, count_cut_off, mapping_path)
