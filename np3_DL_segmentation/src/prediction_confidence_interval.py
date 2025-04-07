# https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.stats.bootstrap.html
# article - Confidence intervals for performance estimates in 3D medical image segmentation-  https://arxiv.org/pdf/2307.10926
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import pandas as pd
from os import path, makedirs
import sys


# computes the statistics and confidence interval bootstrap for a DL metric
# stored by the save_prediction procedure while testing,
# uses a csv with the predictions by entry and by row, with the variables of interest by column
def compute_variables_stats_ci_bootstrap(entries_pred_file, output_path, output_suffix,
                                         variables_interest=["mIoU"], confidence_interval_level=0.95,
                                         plot_variable_distribution = True,
                                         plot_ci_distribution = True):
    entries_pred = pd.read_csv(entries_pred_file)
    res_vars_interest = pd.DataFrame(columns=["variable_name", "mean", "std", "ci_error_low",
                                              "ci_error_high", "ci_low", "ci_high", "ci_std"])
    rng = np.random.default_rng()

    variables_interest = entries_pred.columns.intersection(variables_interest).values
    if len(variables_interest) == 0:
        sys.exit("None of the informed variables of interest appear in the entries table columns. "
                 "Check if the informed names match the columns names present in the informed table and retry.")
    print("\nValid variables names to be analysed:", ", ".join(variables_interest),"\n")

    # create output dir
    if not path.exists(output_path):
        makedirs(output_path)

    for i, var_interest in enumerate(variables_interest):
        # store the results for the variable of interest - statistics and confidence interval
        # will store the variable_name, mean, std, ci_error_low, ci_error_high, ci_low, ci_high, ci_std
        res_var_stats_ci = []
        #
        res_var_stats_ci.append(var_interest)
        res_var_stats_ci.append(entries_pred[var_interest].mean())
        res_var_stats_ci.append(entries_pred[var_interest].std())
        # print statistics
        print("\n* Statistics for",var_interest)
        print("Data mean: ",str(res_var_stats_ci[1]))
        print("Data std: ",str(res_var_stats_ci[2]))
        #
        if plot_variable_distribution:
            if not path.exists(path.join(output_path, "vars_distribution"+"_"+output_suffix)):
                makedirs(path.join(output_path, "vars_distribution"+"_"+output_suffix))
            fig, ax = plt.subplots()
            entries_pred[var_interest].plot.density(ax=ax)
            entries_pred[var_interest].hist(density=True, bins=25,ax=ax, grid=True)
            ax.set_title('Histogram with density for metric ' + var_interest)
            ax.set_xlabel(var_interest)
            fig.savefig(path.join(output_path, "vars_distribution"+"_"+output_suffix, "hist_"+var_interest+"_"+output_suffix+".png"))
            plt.close()
            #plt.show()
        #
        # compute confidence interval and print result
        data = entries_pred[var_interest][entries_pred[var_interest].notna()].values
        res = bootstrap((data,), np.std, confidence_level=confidence_interval_level, random_state=rng, n_resamples=15000,
                        method="BCa")
        res_var_stats_ci.append(res.confidence_interval.low)
        res_var_stats_ci.append(res.confidence_interval.high)
        res_var_stats_ci.append(res_var_stats_ci[1]-res.confidence_interval.low)
        res_var_stats_ci.append(res_var_stats_ci[1]+res.confidence_interval.high)
        res_var_stats_ci.append(res.standard_error)
        print("Confidence interval of", confidence_interval_level,"low and high error: ", res_var_stats_ci[3],",",
              res_var_stats_ci[4])
        print("Confidence interval of", confidence_interval_level,": ",
              res_var_stats_ci[5],",",
              res_var_stats_ci[6])
        print("Confidence interval standard error: ", res_var_stats_ci[7])
        #
        if plot_ci_distribution:
            if not path.exists(path.join(output_path, "vars_ci_bootstrap_distribution"+"_"+output_suffix)):
                makedirs(path.join(output_path, "vars_ci_bootstrap_distribution"+"_"+output_suffix))
            fig, ax = plt.subplots()
            ax.hist(res.bootstrap_distribution, bins=25, density=True)
            ax.set_title('Bootstrap Distribution for '+var_interest)
            ax.set_xlabel('statistic value')
            ax.set_ylabel('frequency')
            fig.savefig(path.join(output_path, "vars_ci_bootstrap_distribution"+"_"+output_suffix,
                                  "boot_dist_" + var_interest + "_"+output_suffix+ ".png"))
            plt.close()
            #plt.show()
        # add the variable information to the results dataframe
        res_vars_interest.loc[i,:] = res_var_stats_ci

    # store final result, stats and ci of variables of interest
    res_vars_interest.to_csv(path.join(output_path,
                                       "res_stats_ci_bootstrap_"+output_suffix+".csv"))


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        entries_pred_file = sys.argv[1]
        output_path = sys.argv[2]
        variables_interest = ["mIoU"]
        if len(sys.argv) >= 4:
            variables_interest = sys.argv[3].split(",")
        confidence_interval_level = 0.95
        if len(sys.argv) >= 5:
            confidence_interval_level = float(sys.argv[4])
        plot_variable_distribution = True
        if len(sys.argv) >= 6:
            plot_variable_distribution = bool(sys.argv[5])
        plot_ci_distribution = True
        if len(sys.argv) >= 7:
            plot_ci_distribution = bool(sys.argv[6])
        output_suffix = ""
        if len(sys.argv) >= 8:
            output_suffix = sys.argv[7]
    else:
        sys.exit("Wrong number of parameters. At least two parameters must be supplied to compute the stats and "
                 "confidence interval bootstrap for a DL metric prediction. \n"
                 "1. entries_pred_file: path to the prediciton output file with one variable by column and one entry by row;\n"
                 "2. output_path: the path to save the output statistics and ci result;\n"
                 "3. variables_interest: the names of the variables (columns) to compute the stats and ci, separated by coma without spaces (default: 'mIoU');\n"
                 "4. confidence_interval_level: the level of confidence interval to be used in the bootstrap computation (default: 0.95);\n"
                 "5. plot_variable_distribution: a boolean to decide to plot the variables histogram and save to the output path in the 'vars_distribution' folder (default: True);\n"
                 "6. plot_ci_distribution: a boolean to decide to plot the variables ci density distribution and save to the output path in the 'vars_ci_bootstrap_distribution' folder (default: True);\n"
                 "7. output_suffix: a suffix to name the output file - useful to refer to the metric name and job name (default: '').\n")

    compute_variables_stats_ci_bootstrap(entries_pred_file, output_path,
                                         variables_interest=variables_interest, confidence_interval_level=confidence_interval_level,
                                         plot_variable_distribution=plot_variable_distribution,
                                         plot_ci_distribution=plot_ci_distribution, output_suffix=output_suffix)

