import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.pylab as pylab
from pathlib import Path


def str2flist(l):
    return [float(i) for i in l.split(',')]

# matplotlib palletes
colors_set1 = np.asarray(['0.97,0.51,0.75', '0.60,0.60,0.60', '0.65,0.34,0.16', '1.00,1.00,0.20', '1.00,0.50,0.00',
               '0.60,0.31,0.64', '0.30,0.69,0.29', '0.22,0.49,0.72', '0.89,0.10,0.11'])
colors_tab21 = np.asarray(['0.12,0.47,0.71', '0.60,0.60,0.60', '0.68,0.78,0.91', '1.00,0.50,0.05', '1.00,0.73,0.47',
                '0.17,0.63,0.17',
                '0.60,0.87,0.54', '0.84,0.15,0.16', '1.00,0.60,0.59', '0.58,0.40,0.74', '0.77,0.69,0.84',
                '0.55,0.34,0.29', '0.77,0.61,0.58', '0.89,0.47,0.76', '0.97,0.71,0.82', '0.50,0.50,0.50',
                '0.78,0.78,0.78', '0.74,0.74,0.13', '0.86,0.86,0.55', '0.09,0.75,0.81', '0.62,0.85,0.90'])

# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar/49357445#49357445
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed

def plot_learning_curves(curves_path, curves_names, curves_colors, curves_steps_by_epoch, curves_style, max_epoch, output_img_path,
                         curves_title, curves_xlabel, curves_ylabel, smooth_rate=0.0, figsize=(8, 6),legend_ncol=1, ylim = None):
    # read the learning curves
    curves_data = [pd.read_csv(curve_path) for curve_path in curves_path]
    for i, curve in enumerate(curves_data):
        if not curve.columns.str.contains("epoch").any():
            curve['epoch'] = curve.Step / curves_steps_by_epoch[i]
    # set max_epoch limit in the data for better plotting
    curves_data = [curve.loc[curve.epoch <= max_epoch, :] for curve in curves_data]
    # set styling
    mpl.style.use('seaborn')
    params = {'legend.fontsize': 'xx-large',
              'figure.figsize': figsize,
              'axes.labelsize': 'xx-large',
              'axes.titlesize':'xx-large',
              'xtick.labelsize':'xx-large',
              'ytick.labelsize':'xx-large'}
    pylab.rcParams.update(params)
    plt.subplots(1)
    # plot the curves
    for i in range(len(curves_data)):
        plt.plot(curves_data[i].epoch, smooth(curves_data[i].Value, smooth_rate), ls=curves_style[i],
                 color=str2flist(curves_colors[i]),  label=curves_names[i])
    # set the plot title and labels
    plt.title(curves_title)
    plt.xlabel(curves_xlabel), plt.ylabel(curves_ylabel), plt.legend(loc="best",frameon=True,ncol=legend_ncol)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    # save figure and show plot
    plt.savefig(output_img_path)
    plt.show()


#####################################
# PLOT SYSTEMATIC ANALYSIS - IMAGE TYPE
####################################
output_img_path = "modelAtomC347CA56/img-type/plot_data/run-img_types_mIoU.png"
curves_path = sorted(list(Path("modelAtomC347CA56/img-type/plot_data/mIoU/").rglob("*.csv")))
curves_names = ["V1 qRankMask_5", "V2 qRank0.5", "V3 qRank0.7", "V4 qRank0.75", "V5 qRank0.8", "V6 qRank0.85",
                "V7 qRank0.9", "V8 qRank0.95", "V9 qRankMask"]
curves_steps_by_epoch = [4552]*10
curves_style = ['-']*10
curves_colors = colors_tab21
curves_title = "mIoU - Train and Validation"
curves_xlabel= "Epochs"
curves_ylabel= "mean IoU (%)"
max_epoch = 120
plot_learning_curves(curves_path, curves_names, curves_colors, curves_steps_by_epoch, curves_style, max_epoch,
                     output_img_path, curves_title, curves_xlabel, curves_ylabel, figsize=(9, 7), legend_ncol=2)
# Loss val
output_img_path = "modelAtomC347CA56/img-type/plot_data/runs_val_loss.png"
curves_path = sorted(list(Path("modelAtomC347CA56/img-type/plot_data/loss/").rglob("*.csv")))
# curves_path = [curves_path[i] for i in curves_order]
curves_title = "Loss Function wSL - Validation"
curves_ylabel= "Loss"
plot_learning_curves(curves_path, curves_names, curves_colors, curves_steps_by_epoch, curves_style,
                     max_epoch, output_img_path, curves_title, curves_xlabel, curves_ylabel, figsize=(9, 7), legend_ncol=2)

# IoU Val
for c in ['Atom', 'Background', 'C3', 'C4', 'C5', 'C6', 'C7', 'CA5', 'CA6']:
    output_img_path = "modelAtomC347CA56/img-type/plot_data/IoU_val/runs_IoU_"+c+".png"
    curves_path = sorted(list(Path("modelAtomC347CA56/img-type/plot_data/IoU_val/"+c).rglob("*.csv")))
    curves_title = "IoU "+c+" - Validation"
    curves_ylabel = "IoU %"
    plot_learning_curves(curves_path, curves_names, curves_colors, curves_steps_by_epoch, curves_style,
                         max_epoch, output_img_path, curves_title, curves_xlabel, curves_ylabel, figsize=(9, 7), legend_ncol=2, smooth_rate=0.4)

