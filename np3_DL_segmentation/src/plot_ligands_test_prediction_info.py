# script to plot the ligands test prediction result against the entries info present in a provided table
# the test predictions are parsed from a list of predictions dirs (different kfolds) provided by the user
# the metrics IoU, F1-Score, Recall and Precision are extracted from the prediction dirs, concatenated and
# used to compute the average score of each metric, which is used in the plots
# the classes names of the prediction result are informed by the user
# the script outputs the plots to the provided output path together with the final table used in the plots,
# named as "ligands_prediction_info_"+output_name+".csv"
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# concatenate metrics data from different kfold results
# set result table names with test prediction
f1_recall_precision_results_name = "entries_f1_recall_precision.csv"
mIou_result_name = "entries_ious.csv"
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
#          'tab:gray', 'tab:olive', 'tab:cyan']

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def violin_plot_ligs_pred_info(ligs_data, metric_name, info_name, data_name, output_path, cut_size=0.1,
                               vcolor='tab:blue'):
    y = ligs_data[metric_name].values
    x = ligs_data[info_name].values
    # group the info by cut_size A
    cuts = np.arange(x.min().round(1), x.max().round(1), cut_size)
    dataset_cuts = []
    valid_cuts = []
    # only add a new dataset if any value is inside the given range
    for i, cut in enumerate(cuts):
        if ((x >= cut) & (x < cut + cut_size)).any():
            dataset_cuts.append(np.sort(y[np.where((x >= cut) & (x < cut + cut_size))[0]]))
            valid_cuts.append(i)
    # filter valid cuts, which had at least one value
    cuts = cuts[valid_cuts]
    # Create the scatter plot
    vp = plt.violinplot(dataset_cuts,
                        showmeans=False,
                        showmedians=False)
    # Access the 'cbars' (center vertical lines) and set their color
    vp['cbars'].set_color('black')
    # Access 'cmaxes' (top whiskers) and 'cmins' (bottom whiskers) and set their color/style
    vp['cmaxes'].set_color('black')
    vp['cmins'].set_color('black')
    # vp['cmedians'].set_linestyle(':')  # You can also change the line style
    #vp['cmedians'].set_color('tab:orange')
    # Set the face color for each violin body
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(vcolor)
        body.set_edgecolor('black')
        body.set_alpha(1)
    # add interquantile rectangle in black and median value as white circle
    percentiles = [np.percentile(dataset_cuts[i], [25, 50, 75], axis=0) for i in range(len(dataset_cuts))]
    quartile1, medians, quartile3 = [], [], []
    for i in range(len(percentiles)):
        quartile1.append(percentiles[i][0])
        medians.append(percentiles[i][1])
        quartile3.append(percentiles[i][2])
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(dataset_cuts, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    # Add title and labels
    plt.title(metric_name + " x " + info_name + " for " + data_name)
    plt.xlabel(info_name)
    plt.ylabel(metric_name)
    # set xticks
    plt.xticks(np.arange(1.0, float(len(cuts) + 1), 1.0),
               ["[" + str(cut.round(1)) + "-" + str((cut + cut_size).round(1)) + ")" for cut in cuts], rotation=45)
    plt.tight_layout()
    # save figure and show plot
    if not (output_path / info_name).exists():
        (output_path / info_name).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / info_name / ("violinplot_" + info_name + "_" + metric_name + ".png"))
    # Display the plot
    #plt.show()
    plt.close()


## func
def extract_ligand_prediction_info_plot(ligs_dataset_path, prediction_dirs_input, classes_names_input, data_name,
                                        output_path, output_name):
    output_path = Path(output_path)
    if not (output_path).exists():
        (output_path).mkdir(parents=True, exist_ok=True)
    # convert predictions dirs to list
    prediction_dirs = [Path(pred_dir) for pred_dir in prediction_dirs_input.split(';')]
    # convert classes names to list
    classes_names = classes_names_input.split(',')
    
    # read metrics prediction results and compute average value by metric and merge in a single result table
    f1_recall_precision_results = pd.concat([pd.read_csv(pred_dir/f1_recall_precision_results_name) for pred_dir in prediction_dirs])
    f1_recall_precision_results["mF1"] = f1_recall_precision_results[[class_name+"_f1" for class_name in classes_names]].mean(1)*100.0
    f1_recall_precision_results["mRecall"] = f1_recall_precision_results[[class_name+"_recall" for class_name in classes_names]].mean(1)*100.0
    f1_recall_precision_results["mPrecision"] = f1_recall_precision_results[[class_name+"_precision" for class_name in classes_names]].mean(1)*100.0
    f1_recall_precision_results = f1_recall_precision_results[['ligID','mF1','mRecall','mPrecision']]
    
    mIoU_results = pd.concat([pd.read_csv(pred_dir/mIou_result_name) for pred_dir in prediction_dirs])
    mIoU_results["mIoU"] = mIoU_results[classes_names].mean(1)
    mIoU_results = mIoU_results[['ligID','mIoU']]
    
    prediction_results = mIoU_results.merge(f1_recall_precision_results)
    
    if prediction_results.ligID.duplicated().any():
        print("WARNING: there are duplicated ligand entries (ligIDs). Check the inputs if this is not wanted.")
        
    # read the ligands training dataset and merge with the metrics prediction avg result
    ligs = pd.read_csv(ligs_dataset_path)
    ligs = ligs[["ligID", "ligCode", "RefinementResolution","bfactor","bfactor_std","AverageBFactor","bfactor_ratio",
                 "numAtoms","point_cloud_size_qRankMask",'point_cloud_size_qRank0.95']]
    ligs = ligs.merge(prediction_results)
    ligs.to_csv(output_path/("ligands_prediction_info_"+output_name+".csv"), index=False)
    
    # now make plots by different infos
    metrics_average_name = ["mIoU","mF1","mRecall","mPrecision"]
    
    info_name = "RefinementResolution"
    for metric_name in metrics_average_name:
        violin_plot_ligs_pred_info(ligs, metric_name, info_name, data_name, output_path, cut_size=0.1, vcolor='tab:blue')
    
    info_name = "bfactor"
    for metric_name in metrics_average_name:
        violin_plot_ligs_pred_info(ligs, metric_name, info_name, data_name, output_path, cut_size=10, vcolor='tab:green')
    
    info_name = "bfactor_std"
    for metric_name in metrics_average_name:
        violin_plot_ligs_pred_info(ligs, metric_name, info_name, data_name, output_path, cut_size=5, vcolor='tab:brown')
    
    info_name = "AverageBFactor"
    for metric_name in metrics_average_name:
        violin_plot_ligs_pred_info(ligs, metric_name, info_name, data_name, output_path, cut_size=5, vcolor='tab:olive')
    
    info_name = "bfactor_ratio"
    for metric_name in metrics_average_name:
        violin_plot_ligs_pred_info(ligs, metric_name, info_name, data_name, output_path, cut_size=0.75, vcolor='tab:cyan')
    
    info_name = "numAtoms"
    for metric_name in metrics_average_name:
        violin_plot_ligs_pred_info(ligs, metric_name, info_name, data_name, output_path, cut_size=10, vcolor='tab:red')
    
    info_name = "point_cloud_size_qRankMask"
    for metric_name in metrics_average_name:
        violin_plot_ligs_pred_info(ligs, metric_name, info_name, data_name, output_path, cut_size=500, vcolor='tab:pink')
    
    info_name = 'point_cloud_size_qRank0.95'
    for metric_name in metrics_average_name:
        violin_plot_ligs_pred_info(ligs, metric_name, info_name, data_name, output_path, cut_size=150, vcolor='tab:purple')


if __name__ == "__main__":
    if len(sys.argv) >= 6:
        ligs_dataset_path = sys.argv[1]
        prediction_dirs_input = sys.argv[2]
        classes_names_input = sys.argv[3]
        data_name = sys.argv[4]
        output_path = sys.argv[5]
        output_name = sys.argv[6]
    else:
        sys.exit("Wrong number of arguments. Six arguments must be supplied in order to extract the ligands prediction "
                 "results and merge with their info table, and then plot the relevant information and store to the "
                 "desired output. Parameters: \n"
                 "  1. The path to the CSV file containing the ligands training dataset with the list of ligands used "
                 "for training. Mandatory columns are: 'ligID', 'RefinementResolution','bfactor','bfactor_std',"
                 "'AverageBFactor','bfactor_ratio','numAtoms','point_cloud_size_qRankMask','point_cloud_size_qRank0.95'. "
                 "This file is expected to be the one used for obtaining the prediction results. \n"
                 "  2. One or more path to a prediction result separated by ';'. The results of all provided prediction "
                 "directories will be concatenated, useful to join different kfold testing result. \n"
                 "  3. The names of all the classes present in the prediction result, separated by comma ','. "
                 "The average of all metrics will be computed based on this list of classes.\n"
                 "  4. The name of the data being analysed, this will be used in the plots title to identify the data "
                 "being used.\n"
                 "  5. The path to the output directory where the plots and final table must be stored. "
                 "The plost will be organized in subplots by the name of the information being ploted. "
                 "The final table will contain the metrics average values and used information by ligID.\n "
                 "  6. The output name to be used in the final table file name following the format: "
                 "'ligands_prediction_info_'+output_name+'.csv'.\n\n"
                 )
    extract_ligand_prediction_info_plot(ligs_dataset_path, prediction_dirs_input, classes_names_input, data_name,
                                        output_path, output_name)
