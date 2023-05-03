import argparse
from time import ctime


def str2opt(arg):
    assert arg in ['SGD', 'Adam']
    return arg


def str2scheduler(arg):
    assert arg in ['StepLR', 'PolyLR', 'ExpLR', 'SquaredLR']
    return arg


def str2bool(v):
    return v.lower() in ('true', '1', 't')


def str2list(l):
    return [int(i) for i in l.split(',')]

def str2flist(l):
    return [float(i) for i in l.split(',')]


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


arg_lists = []
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='MinkUNet34C_CONVATROUS_HYBRID', help='Model name',
                     choices=['MinkUNet101', 'MinkUNet14', 'MinkUNet14A', 'MinkUNet14B', 'MinkUNet14C', 'MinkUNet14D',
                              'MinkUNet18', 'MinkUNet18A', 'MinkUNet18B', 'MinkUNet18D', 'MinkUNet34', 'MinkUNet34A',
                              'MinkUNet34B', 'MinkUNet34C', 'MinkUNet34CIN', 'MinkUNet50','Res16UNet34CIN',
                              'Res16UNet34C', 'MinkUNet34CIN_CONVATROUS_HYBRID', 'MinkUNet34C_CONVATROUS_HYBRID'])
net_arg.add_argument(
    '--conv1_kernel_size', type=int, default=3, help='First layer conv kernel size')
net_arg.add_argument('--weights', type=str, default='None', help='Saved weights to load')
net_arg.add_argument(
    '--weights_for_inner_model',
    type=str2bool,
    default=False,
    help='Weights for model inside a wrapper')
# net_arg.add_argument(
#     '--dilations', type=str2list, default='1,1,1,1', help='Dilations used for ResNet or DenseNet')
# net_arg.add_argument('--dropout', type=float, default=0.5,
#                         help='dropout rate')
# net_arg.add_argument('--emb_dims', type=int, default=1024, metavar='N',
#                     help='Dimension of embeddings')
# net_arg.add_argument('--k', type=int, default=60, metavar='N',
#                     help='Num of nearest neighbors to use')
# parser.add_argument('--num_points', type=int, default=0,
#                         help='num of points to use. When equals 0 disable the sub sampling')
# Wrappers
# net_arg.add_argument('--wrapper_type', default='None', type=str, help='Wrapper on the network')
# net_arg.add_argument(
#     '--wrapper_region_type',
#     default=1,
#     type=int,
#     help='Wrapper connection types 0: hypercube, 1: hypercross, (default: 1)')
# net_arg.add_argument('--wrapper_kernel_size', default=3, type=int, help='Wrapper kernel size')
# net_arg.add_argument(
#     '--wrapper_lr',
#
#     default=1e-1,
#     type=float,
#     help='Used for freezing or using small lr for the base model, freeze if negative')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--lr', type=float, default=2**(-8))
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
# opt_arg.add_argument('--param_histogram_freq', type=int, default=100)
# opt_arg.add_argument('--save_param_histogram', type=str2bool, default=False)
opt_arg.add_argument('--iter_size', type=int, default=1, help='The number of iterations to accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)

# Scheduler
opt_arg.add_argument('--scheduler', type=str2scheduler, default='StepLR')
opt_arg.add_argument('--max_iter', type=int, default=1e6)
opt_arg.add_argument('--step_size', type=int, default=200, help="Number of steps in epochs")
opt_arg.add_argument('--step_gamma', type=float, default=0.1)
opt_arg.add_argument('--poly_power', type=float, default=0.9)
opt_arg.add_argument('--exp_gamma', type=float, default=0.95)
opt_arg.add_argument('--exp_step_size', type=float, default=445)
opt_arg.add_argument('--max_epoch', type=int, default=100)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--log_dir', type=str, default='outputs/out',
                     help="the output directory where the logging info and the final model will be saved. "
                          "A suffix with the current time will be added to it..")
dir_arg.add_argument('--data_dir', type=str, default='data')

# Data
data_arg = add_argument_group('Data')
# data_arg.add_argument('--dataset', type=str, default='ScannetVoxelization2cmDataset')
# data_arg.add_argument('--temporal_dilation', type=int, default=30)
# data_arg.add_argument('--temporal_numseq', type=int, default=3)
# data_arg.add_argument('--point_lim', type=int, default=-1)
# data_arg.add_argument('--pre_point_lim', type=int, default=-1)
data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--val_batch_size', type=int, default=8)
data_arg.add_argument('--test_batch_size', type=int, default=8)
# data_arg.add_argument('--cache_data', type=str2bool, default=False)
data_arg.add_argument('--num_workers', type=int, default=4, help='num workers for train/test dataloader')
data_arg.add_argument('--num_val_workers', type=int, default=2, help='num workers for val dataloader')
data_arg.add_argument('--ignore_label', type=str2list, default=255)
# data_arg.add_argument('--return_transformation', type=str2bool, default=False)
# data_arg.add_argument('--ignore_duplicate_class', type=str2bool, default=False)
# data_arg.add_argument('--partial_crop', type=float, default=0.)
# data_arg.add_argument('--train_limit_numpoints', type=int, default=0)

data_arg.add_argument('--ligs_data_filepath', type=str, required=True,
                      help='path to a ligands entries table defining the training dataset to be used. '
                           'It must contain the following columns: ligID, entry, kfolds, test_val, grid_space')
data_arg.add_argument('--lig_pcdb_path', type=str, required=True,
                      help='path to the folder where the ligands\' labeled image database in point cloud format is located. It is expected to have a subfolder for each PDB entryID present in the ligands entries table. The PDB entries subfolders should contain the ligand\'s images and labels for all ligID present in that table for each respective entryID. These images will be used for training, validating and testing the model.')
data_arg.add_argument('--pc_type', type=str, default='qRankMask_5',
                      help='the point cloud image type to be used, which correspond to the desired quantile rank contour used to create the image.',
                      choices=['qRankMask', 'qRank0.5', 'qRank0.7', 'qRank0.75', 'qRank0.8',
                               'qRank0.85', 'qRank0.9', 'qRank0.95', 'qRankMask_5'])
                               #'qRankMask_5_75_95', 'qRankMask_5_7_9'])
data_arg.add_argument('--vocab_path', type=str, required=True,
                      help='path to the vocabulary used to label the provided ligands\' images dataset.')
# data_arg.add_argument('--grid_space', type=float, default=0.5,
#                       help='the grid space used to create the ligands point clouds.')
data_arg.add_argument('--kfold', type=int, default=13,
                      help='the k-fold group to be used for testing and validation. '
                           'The other groups will be used for training.')

data_arg.add_argument('--class_mapping_path', type=str, default=None,
                      help='the path to the csv file with the class mapping to be used in the vocabulary simplification. '
                           'Mandatory columns: \'source\', \'mapping\', \'target\'. '
                           'The \'source\' column should contain the provided vocabulary classes index (from 0 to the '
                           'number of classes in the vocabulary, counting background); '
                           'the \'mapping\' column should contain the new classes names; '
                           'and the \'target\' column should contain the new classes index, starting in 0 and '
                           'increasing to the number of new classes. '
                           'The order of the \'source\' column values should be the same of the vocabulary classes and '
                           'the last class *must* be the background class.')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='set to False for testing')
train_arg.add_argument('--log_freq', type=int, default=3000, help='statistics logging frequency in number of steps')
# train_arg.add_argument('--test_log_freq', type=int, default=100, help='print frequency')
train_arg.add_argument('--save_freq', type=int, default=1000, help='checkpoint save frequency in terms of steps; also used to log_every_n_steps in the Trainer setup in pytorch-lightning')
train_arg.add_argument('--val_freq', type=int, default=5000, help='validation frequency in number of steps')
train_arg.add_argument(
    '--empty_cache_freq', type=int, default=1, help='Clear pytorch cache frequency in number of steps')
# train_arg.add_argument('--train_phase', type=str, default='train', help='Dataset for training')
# train_arg.add_argument('--val_phase', type=str, default='val', help='Dataset for validation')
# train_arg.add_argument('--overwrite_weights', type=str2bool, default=True, help='Overwrite checkpoint during training')
train_arg.add_argument(
    '--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
# train_arg.add_argument(
#     '--resume_optimizer',
#     default=True,
#     type=str2bool,
#     help='Use checkpoint optimizer states when resume training')
# train_arg.add_argument('--eval_upsample', type=str2bool, default=False)
train_arg.add_argument(
    '--lenient_weight_loading',
    type=str2bool,
    default=False,
    help='Weights with the same size will be loaded')
# train_arg.add_argument(
#     '--loss_weight_H_ratio',
#     type=str2bool,
#     default=False,
#     help='Initialize the weights of the CrossEntropyLoss function equals to the class representation imbalance ratio')
train_arg.add_argument(
    '--loss_weights',
    type=str2flist,
    default='1',
    help='Set the class weights in the Loss function equals to the provided list, separated by comma. For no weights, '
         'set as 1 (default). It should follow the order of the vocabulary classes or mapping.')
# train_arg.add_argument(
#     '--stochastic_weight_avg',
#     type=str2bool,
#     default=False,
#     help='Apply the Stochastic Weight Averaging (SWA), which can make your models generalize better at virtually no additional cost. This can be used with both non-trained and trained models. The SWA procedure smooths the loss landscape thus making it harder to end up in a local minimum during optimization.')

train_arg.add_argument(
    '--loss_func',
    type=str,
    default='SL',
    help='Selects the desired loss function: Cross Entropy Loss (CE) or Symmetric Cross entropy Learning (SL)',
    choices = ['CE', 'SL']
)
train_arg.add_argument(
    '--SL_alpha',
    type=float,
    default=0.1,
    help='The alpha parameter for the Symmetric Cross entropy Learning (SL)'
)
train_arg.add_argument(
    '--SL_beta',
    type=float,
    default=5,
    help='The beta parameter for the Symmetric Cross entropy Learning (SL)'
)

train_arg.add_argument(
    '--rotation_rate',
    type=float,
    default=0.5,
    help='Random rotation rate to be applied to the train dataset during training (augmentation). '
         'It random rotates the training input point cloud in the 3 axis')

# Data augmentation
# data_aug_arg = add_argument_group('DataAugmentation')
# data_aug_arg.add_argument(
#     '--use_feat_aug', type=str2bool, default=True, help='Simple feat augmentation')
# data_aug_arg.add_argument(
#     '--data_aug_color_trans_ratio', type=float, default=0.10, help='Color translation range')
# data_aug_arg.add_argument(
#     '--data_aug_color_jitter_std', type=float, default=0.05, help='STD of color jitter')
# data_aug_arg.add_argument('--normalize_color', type=str2bool, default=True)
# data_aug_arg.add_argument('--data_aug_scale_min', type=float, default=0.9)
# data_aug_arg.add_argument('--data_aug_scale_max', type=float, default=1.1)
# data_aug_arg.add_argument(
#     '--data_aug_hue_max', type=float, default=0.5, help='Hue translation range. [0, 1]')
# data_aug_arg.add_argument(
#     '--data_aug_saturation_max',
#     type=float,
#     default=0.20,
#     help='Saturation translation range, [0, 1]')

# Test
test_arg = add_argument_group('Test')
# test_arg.add_argument('--visualize', type=str2bool, default=False)
# test_arg.add_argument('--test_temporal_average', type=str2bool, default=False)
# test_arg.add_argument('--visualize_path', type=str, default='outputs/visualize')
test_arg.add_argument('--save_prediction', type=str2bool, default=False)
test_arg.add_argument('--save_pred_dir', type=str, default='outputs/pred')


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)
# misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--num_gpu', type=int, default=1, help="Number of GPU devices to be used. "
                                                             "If > 1 multi-GPU is enabled when possible, "
                                                             "else single GPU is used. Ignored if is_cuda is False.")
misc_arg.add_argument('--gpu_index', type=str2list, default=[0], help="GPU index to be used when is_cuda is True and "
                                                               "num_gpu >= 1. The indexes when num_gpu > 1 must be comma separated.")
misc_arg.add_argument('--seed', type=int, default=123)


def get_config():
    config = parser.parse_args()
    # if config.resume:
    #     config.log_dir = config.resume
    # else:
    config.log_dir = config.log_dir + "_" + ('train' if config.is_train else 'test') + "_" + config.pc_type + "_kfold_" + \
                     str(config.kfold) + "_model-" + config.model + "_" + \
                     str(ctime().replace("  ", "_").replace(" ", "_").replace(":", "-"))
    return config  # Training settings
