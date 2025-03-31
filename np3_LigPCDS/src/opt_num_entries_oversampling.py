from gurobipy import *
import pandas as pd
import numpy as np
from vocabulary_SP_classes_fromSDF_statistics import ligs_class_freq_stats, save_class_stats_plot
from pathlib import Path
import sys

ligs_data_path = Path("")
vocab_path = ""
num_ligCode_limits = [0,500]
num_entries_by_class_limits = [10000, 15000]

def oversample_opt_ligands_dataset(vocab_path, ligs_data_path, mapping_path, min_num_ligCode, max_num_ligCode,
                                   min_num_entries_by_class,
                                   max_num_entries_by_class, zero_total = False):
    # read the classes
    vocab = np.asarray([line.rstrip('\n') for line in open(vocab_path)])
    # list the vocab column names
    classes_col = np.asarray([str(i) for i in range(len(vocab))])

    ligs_data_path = Path(ligs_data_path)

    # read the ligs data
    ligs_data = pd.read_csv(ligs_data_path)
    # count number of ligCodes
    ligs_data['total'] = 1
    ligs_count = ligs_data.groupby('ligCode').agg({'total': sum})
    # remove duplicated ligCode keep the one with no missing heavy atoms
    ligs_data = ligs_data.sort_values('missingHeavyAtoms').reset_index(drop=True)
    ligs_data = ligs_data[~ligs_data.ligCode.duplicated()]

    # filter only the ligCode and the classes labels columns
    ligs_data = ligs_data[['ligCode']+list(classes_col)]
    ligs_data['total'] = [ligs_count.total[ligCode] for ligCode in ligs_data.ligCode]
    del ligs_count

    # apply mapping before filtering the classes
    if mapping_path:
        mapping_path = Path(mapping_path)
        mapping = pd.read_csv(mapping_path).sort_values('source')
        mapping = mapping.loc[0:vocab.shape[0] - 1, ]  # remove background
        new_vocab = mapping.target.unique()
        new_vocab.sort()
        new_vocab = pd.DataFrame(new_vocab)
        vocab0_cols = [ligs_data.columns.get_loc(str(x)) for x in list(range(len(vocab)))]
        dist_old_vocab = ligs_data.iloc[:,vocab0_cols]
        ligs_data = ligs_data.drop(ligs_data.columns[vocab0_cols], axis=1)
        # aggregate the distribution of the new vocab based on the mapping and save to the ligands entries file
        for new_label_idx in new_vocab[0]:
            ligs_data[str(new_label_idx)] = dist_old_vocab[
                mapping.source[mapping.target == new_label_idx].values.astype(str)].sum(1)
        # replace old vocab with new
        vocab = pd.DataFrame({0: mapping[~mapping[['mapping', 'target']].duplicated()].sort_values('target').mapping.reset_index(drop=True)})
        vocab = vocab[0].values
        del dist_old_vocab
        # list the vocab column names
        classes_col = np.asarray([str(i) for i in new_vocab[0]])
    else:
        # list the vocab column names
        classes_col = np.asarray([str(i) for i in range(len(vocab))])
    # filter the classes that have no entries
    rm_no_entry_classes = ((ligs_data[classes_col] >0).sum() > 0).values
    classes_col = classes_col[rm_no_entry_classes]
    vocab = vocab[rm_no_entry_classes]
    del rm_no_entry_classes

    ligs_data = ligs_data[['ligCode', 'total']+list(classes_col)].reset_index(drop=True)
    print("Number of lig codes:", ligs_data.shape[0])
    print("Current total number of entries by class:\n",
          (ligs_data[classes_col] > 0).multiply(ligs_data['total'], axis="index").sum())
    print("Maximum number of entries of the classes:", max((ligs_data[classes_col] > 0).multiply(ligs_data['total'], axis="index").sum()),
          "\nMinimum number of entries of the classes:", min((ligs_data[classes_col] > 0).multiply(ligs_data['total'], axis="index").sum()),
          "\n Ratio:", max((ligs_data[classes_col] > 0).multiply(ligs_data['total'], axis="index").sum()) /
          min((ligs_data[classes_col] > 0).multiply(ligs_data['total'], axis="index").sum()))

    print("Current total number of atoms by class:\n",
          ligs_data[classes_col].multiply(ligs_data['total'], axis="index").sum())
    print("Maximum number of atoms of the classes", max(ligs_data[classes_col].multiply(ligs_data['total'], axis="index").sum()),
          "\nMinimum number of atoms of the classes", min(ligs_data[classes_col].multiply(ligs_data['total'], axis="index").sum()),
          "\n Ratio:", max(ligs_data[classes_col].multiply(ligs_data['total'], axis="index").sum()) /
          min(ligs_data[classes_col].multiply(ligs_data['total'], axis="index").sum()))

    print("\n ** Start modeling the best oversampling of the data set **")


    # set the model variables value
    num_ligCode_limits = [min_num_ligCode, max_num_ligCode]
    num_entries_by_class_limits = [min_num_entries_by_class, max_num_entries_by_class]
    # max_num_points_by_class = 1000000
    print("* Limits of the number of new entries by ligand code:", num_ligCode_limits)
    print("* Limits of the number of new entries by class:", num_entries_by_class_limits, "\n\n")

    # set the total of each lig Code to zero
    if zero_total:
        save_total = ligs_data[["total"]]
        ligs_data.loc[:, "total"] = 0

    model = Model('Ligand Dataset Oversampling Equal Class distribution')

    # add the number of ligand by code variable
    num_ligCode_var = model.addVars(list(ligs_data.ligCode), name="num_ligCode", lb=0)
    # num_ligCode_var = model.addVars(ligCodes, name="num_ligCode", lb=min_num_ligCode)
    # num_ligCode_var = model.addVars(ligCodes, name="num_ligCode",  ub=max_num_ligCode)
    num_entries_by_class_var = model.addVars(list(classes_col), name="num_entries_by_class", lb=num_entries_by_class_limits[0],
                                             ub = num_entries_by_class_limits[1])
    num_atoms_by_class_var = model.addVars(list(classes_col), name="num_atoms_by_class")


    # add constrain min max number of ligands by code (current + created)
    model.addConstrs(((num_ligCode_var[ligCode] + int(ligs_data.loc[ligs_data.ligCode == ligCode, "total"])) >= num_ligCode_limits[0]
                      for ligCode in num_ligCode_var), name = "min_number_entries_by_class")
    model.addConstrs(((num_ligCode_var[ligCode] + int(ligs_data.loc[ligs_data.ligCode == ligCode, "total"])) <= num_ligCode_limits[1]
                      for ligCode in num_ligCode_var), name = "max_number_entries_by_class")

    # add the constrain to make the total number of entries by class between the num_entries_by_class range
    model.addConstrs((quicksum(int(ligs_data.loc[ligs_data.ligCode == ligCode, class_col] > 0) *
                                   (num_ligCode_var[ligCode] + int(ligs_data.loc[ligs_data.ligCode == ligCode, "total"]))
                                   for ligCode in num_ligCode_var) >= num_entries_by_class_var[class_col]
                          for class_col in classes_col), name = "min_number_entries_by_class")
    model.addConstrs((quicksum(int(ligs_data.loc[ligs_data.ligCode == ligCode, class_col] > 0) *
                               (num_ligCode_var[ligCode] + int(ligs_data.loc[ligs_data.ligCode == ligCode, "total"]))
                               for ligCode in num_ligCode_var) <= num_entries_by_class_var[class_col]
                      for class_col in classes_col), name = "max_number_entries_by_class")

    # add the constrain to limit the total number of atoms by class and then minimize the limit
    model.addConstrs((quicksum(ligs_data.loc[ligs_data.ligCode == ligCode, class_col].values[0] *
                                   (num_ligCode_var[ligCode] + int(ligs_data.loc[ligs_data.ligCode == ligCode, "total"]))
                                   for ligCode in num_ligCode_var) >= num_atoms_by_class_var[class_col]
                          for class_col in classes_col), name = "min_number_points_by_class")
    model.addConstrs((quicksum(ligs_data.loc[ligs_data.ligCode == ligCode, class_col].values[0] *
                               (num_ligCode_var[ligCode] + int(ligs_data.loc[ligs_data.ligCode == ligCode, "total"]))
                               for ligCode in num_ligCode_var) <= num_atoms_by_class_var[class_col]
                      for class_col in classes_col), name = "max_number_points_by_class")


    obj = quicksum(num_entries_by_class_var[class_col] + num_atoms_by_class_var[class_col]
                      for class_col in classes_col)
    model.setObjective(obj, GRB.MINIMIZE)
    # set object to maximize the total number of atoms of the less abundant class = classes with total points < than 2/3 of the maximum
    # total_atoms_by_class = ligs_data[classes_col].multiply(ligs_data['total'], axis="index").sum()
    # obj = quicksum(
    #     ligs_data.loc[ligs_data.ligCode == ligCode, class_col].values[0] * num_ligCode_var[ligCode]
    #     for class_col in classes_col[total_atoms_by_class < 1/2*total_atoms_by_class.max()]
    #     for ligCode in num_ligCode_var) - quicksum(
    #         ligs_data.loc[ligs_data.ligCode == ligCode, class_col].values[0] * num_ligCode_var[ligCode]
    #         for class_col in classes_col[total_atoms_by_class >= 1/2*total_atoms_by_class.max()]
    #         for ligCode in num_ligCode_var)

    # set object to minimize the total number of points by class
    # obj = quicksum(num_ligCode_var[ligCode] for ligCode in num_ligCode_var)
    # minimize the number of new entries that need to be created
    # obj = quicksum(num_ligCode_var[ligCode]-ligs_data.loc[ligs_data.ligCode == ligCode, 'total'] for ligCode in num_ligCode_var)
    # obj = quicksum(
    #     ligs_data.loc[ligs_data.ligCode == ligCode, class_col].values[0] * num_ligCode_var[ligCode]
    #     for class_col in classes_col
    #     for ligCode in num_ligCode_var)
    # obj = quicksum(
    #     ligs_data.loc[ligs_data.ligCode == ligCode, class_col].values[0] * num_ligCode_var[ligCode]
    #     for class_col in classes_col[(ligs_data[classes_col] > 0).multiply(ligs_data['total'], axis="index").sum() > num_entries_by_class[0]]
    #     for ligCode in num_ligCode_var) + \
    #       quicksum(num_ligCode_var[ligCode]-ligs_data.loc[ligs_data.ligCode == ligCode, 'total'] for ligCode in num_ligCode_var)
    # model.setObjective(obj, GRB.MINIMIZE)

    # model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != 2:
        sys.exit("ERROR model not optimal.")

    # create the output dir
    output_path = ligs_data_path.parent / (
                ligs_data_path.name.replace(".csv", "_aug") + "_ligCode_" + str(num_ligCode_limits[0]) +
                "_" + str(num_ligCode_limits[1]) + "_entriesByClass_" + str(num_entries_by_class_limits[0]) +
                "_" + str(num_entries_by_class_limits[1]) + "_zeroTotal_" + str(zero_total))
    output_path.mkdir(exist_ok=True)

    # restore the total for the comptuations
    if zero_total:
        ligs_data.loc[:, "total"] = save_total

    ligs_data['N'] = [num_ligCode_var[ligCode].X for ligCode in ligs_data.ligCode]
    ligs_data['total_N'] = ligs_data['N']+ligs_data['total']

    print("\n* Total number of entries", (ligs_data['total_N']).sum())
    print("  - Number of entries to be created", (ligs_data['N']).sum())
    print("  - Number of existing entries to be used", (ligs_data['total']).sum())

    print("\n* Total number of entries by class:\n",
          (ligs_data[classes_col] > 0).multiply(ligs_data['total_N'], axis="index").sum())
    print("Maximum number of entries of the classes:", max((ligs_data[classes_col] > 0).multiply(ligs_data['total_N'], axis="index").sum()),
          "\nMinimum number of entries of the classes:", min((ligs_data[classes_col] > 0).multiply(ligs_data['total_N'], axis="index").sum()),
          "\n Ratio:", max((ligs_data[classes_col] > 0).multiply(ligs_data['total_N'], axis="index").sum()) /
          min((ligs_data[classes_col] > 0).multiply(ligs_data['total_N'], axis="index").sum()))

    print("\n* Total number of atoms by class:\n",
          ligs_data[classes_col].multiply(ligs_data['total_N'], axis="index").sum())
    print("Maximum number of atoms of the classes", max(ligs_data[classes_col].multiply(ligs_data['total_N'], axis="index").sum()),
          "\nMinimum number of atoms of the classes", min(ligs_data[classes_col].multiply(ligs_data['total_N'], axis="index").sum()),
          "\n Ratio:", max(ligs_data[classes_col].multiply(ligs_data['total_N'], axis="index").sum()) /
          min(ligs_data[classes_col].multiply(ligs_data['total_N'], axis="index").sum()))


    print("\n* Ligands to be created:")
    print(ligs_data[ligs_data["N"] > 0].sort_values('N', ascending=False))


    ligand_classes_entries = (ligs_data[classes_col] > 0).multiply(ligs_data['total_N'], axis="index")
    ligand_classes_freq_stats_entries = ligs_class_freq_stats(ligand_classes_entries,
                                                              list(range(ligand_classes_entries.shape[1])),
                                                              vocab,
                                                      output_path, "oversample_total_entries")
    save_class_stats_plot(ligand_classes_freq_stats_entries, ligs_data_path, output_path, "all_entries",
                          ['Unique ligand codes occurrences', 'Total ligand entries occurrences'])

    ligand_classes_points = (ligs_data[classes_col]).multiply(ligs_data['total_N'], axis="index")
    ligand_classes_freq_stats_points = ligs_class_freq_stats(ligand_classes_points,
                                                             list(range(ligand_classes_entries.shape[1])),
                                                             vocab,
                                                      output_path, "oversample_total_points")
    save_class_stats_plot(ligand_classes_freq_stats_points, ligs_data_path, output_path)

    ligs_data.to_csv(output_path / ligs_data_path.name.replace(".csv", "_opt_oversampling.csv"))
    model.write((output_path / "model.lp").as_posix())


if __name__ == "__main__":
    zero_total = False
    if len(sys.argv) >= 8:
        ligs_data_path = sys.argv[1]
        vocab_path = sys.argv[2]
        mapping_path = sys.argv[3]
        min_num_ligCode = int(sys.argv[4])
        max_num_ligCode = int(sys.argv[5])
        min_num_entries_by_class = int(sys.argv[6])
        max_num_entries_by_class = int(sys.argv[7])
        if len(sys.argv) >= 9:
            zero_total = (sys.argv[8].upper() == "TRUE")
    else:
        sys.exit("Wrong number of arguments. Six arguments must be supplied in order to compute optimum oversampling of the ligands entries in the"
                 " provided ligands dataset: \n"
                 "  1. The path to the CSV file containing the valid ligands list and their classes frequency. "
                 "This file is expected to be the output of test pc script or the undersampling script. It should be located in the "
                 "'xyz_<ligand csv name>' folder inside the 'ligands' folder, named with the suffix '_class_freq_pc_tested.csv' or "
                 "other file with the vocab classes indices as column names;\n"
                 "  2. The path to the text file containing the vocabulary used to label the ligands. "
                 "It must contain one class per line; \n"
                 "  3. The path to the CSV file containing a mapping between the vocabulary classes and the simplified"
                 "classes, or 'none'. Mandatory columns: source, target. "
                 "The last row must be the mapping for the background class, which is ignored. \n"
                 "  4. The minimum number of ligands by ligCodes that could be created in the oversampling;\n"
                 "  5. The maximum number of ligands by ligCodes that could be created by the oversampling;\n"
                 "  6. The minimum number of ligands entries by class that should be created in the oversampling;\n"
                 "  7. The maximum number of ligands entries by class that should be created by the oversampling;\n"
                 "  8. (optional) Boolean True or False to set the current total ligands by ligCode to zero before the "
                 "model optimization. Default to False.\n"
                 )
    oversample_opt_ligands_dataset(vocab_path, ligs_data_path, mapping_path, min_num_ligCode, max_num_ligCode,
                                   min_num_entries_by_class, max_num_entries_by_class, zero_total)
