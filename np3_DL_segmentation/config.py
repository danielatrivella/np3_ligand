import argparse
from time import ctime


def str2opt(arg):
    assert arg in ['SGD', 'Adam']
    return arg


def str2scheduler(arg):
    assert arg in ['StepLR', 'PolyLR', 'ExpLR', 'SquaredLR', 'CosAnnLR']
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
net_arg.add_argument('--model', type=str, default='MinkUNet34C_CONVATROUS_HYBRID', help='The network model name to be used. The net models that end with IN use instance normalization layer instead of batch normalization.',
                     choices=['MinkUNet101', 'MinkUNet14', 'MinkUNet14A', 'MinkUNet14B', 'MinkUNet14C', 'MinkUNet14D',
                              'MinkUNet18', 'MinkUNet18A', 'MinkUNet18B', 'MinkUNet18D', 'MinkUNet34', 'MinkUNet34A',
                              'MinkUNet34B', 'MinkUNet34C', 'MinkUNet34CIN', 'MinkUNet50','Res16UNet34CIN',
                              'Res16UNet34C', 'MinkUNet34CIN_CONVATROUS_HYBRID', 'MinkUNet34C_CONVATROUS_HYBRID'])
net_arg.add_argument(
    '--conv1_kernel_size', type=int, default=3, help='First layer conv kernel size')
net_arg.add_argument(
    '--resume', default=None, type=str, help='path to latest DL model checkpoint - used to resume from previous training (default: none)')
net_arg.add_argument('--weights', type=str, default='None', help='Saved weights from previous trained DL model to load - used for testing.')
net_arg.add_argument(
    '--lenient_weight_loading',
    type=str2bool,
    default=False,
    help='Weights with the same size will be loaded when weights are provided.')


# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD', help="SGD or Adam")
opt_arg.add_argument('--lr', type=float, default=2**(-8), help="The learning rate.")
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9, help="SGD momentum.")
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1, help="SGD dampening.")
opt_arg.add_argument('--adam_beta1', type=float, default=0.9, help="Adam beta 1 parm.")
opt_arg.add_argument('--adam_beta2', type=float, default=0.999, help="Adam beta 2 parm.")
opt_arg.add_argument('--weight_decay', type=float, default=1e-4, help="optimized weight decay")
opt_arg.add_argument('--iter_size', type=int, default=1, help='The number of iterations to accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05, help="batch norm momentum.")

# Scheduler
opt_arg = add_argument_group('Scheduler')
opt_arg.add_argument('--scheduler', type=str2scheduler, default='StepLR', help="Scheduler name to be used. One of: 'StepLR', 'PolyLR', 'ExpLR', 'SquaredLR', 'CosAnnLR'.")
opt_arg.add_argument('--max_iter', type=int, default=1e6, help="Max iteration")
opt_arg.add_argument('--step_size', type=int, default=200, help="Number of steps in epochs")
opt_arg.add_argument('--step_gamma', type=float, default=0.1, help="Scheduler step gamma parm.")
opt_arg.add_argument('--poly_power', type=float, default=0.9, help="Scheduler poly power parm.")
opt_arg.add_argument('--exp_gamma', type=float, default=0.95, help="Scheduler exp gamma parm.")
opt_arg.add_argument('--exp_step_size', type=float, default=445, help="Scheduler exp step size.")
opt_arg.add_argument('--max_epoch', type=int, default=100, help="Maximum number of epochs to train the current DL model, stops after this number of epochs is reached.")

# Directories
data_arg = add_argument_group('Data In/Out')
data_arg.add_argument('--log_dir', type=str, default='outputs/out',
                     help="the output directory path where the logging info and the final DL model will be saved. "
                          "A suffix will be added to this directory name specifying if it is a train or test output, "
                          "the kfold used, the net model name selected and the current time.")
data_arg.add_argument('--ligs_data_filepath', type=str, required=True,
                      help='path to a ligands entries table defining the training dataset to be used. '
                           'It must contain the following columns: ligID, entry, kfolds, test_val, grid_space')
data_arg.add_argument('--lig_pcds_path', type=str, required=True,
                      help='path to the folder where the LigPCDS with the ligands points clouds are located. '
                           'These point clouds will be used for training, validating and testing a DL model.'
                           'It is expected to have a subfolder for each PDB entryID present in the ligand entries table (ligs_data_filepath). '
                           'The PDB entries subfolders should contain the ligand representations and labels for all ligID '
                           'present in that table for each respective entryID. ')
data_arg.add_argument('--vocab_path', type=str, required=True,
                      help='path to the vocabulary used to label the provided LigPCDS dataset.')
data_arg.add_argument('--class_mapping_path', type=str, default=None,
                      help='the path to the csv file with the class mapping to be used in the vocabulary simplification. '
                           'Mandatory columns: \'source\', \'mapping\', \'target\'. '
                           'The \'source\' column should contain the provided vocabulary classes index (from 0 to the '
                           'number of classes in the vocabulary, counting background); '
                           'the \'mapping\' column should contain the new classes names; '
                           'and the \'target\' column should contain the new classes index, starting in 0 and '
                           'increasing to the number of new classes. '
                           'The order of the \'source\' column values should be the same of the vocabulary classes and '
                           'the last class (defined in the last row) *must* be the background class.')


# Data
hyper_arg = add_argument_group('Hyperparameters')
hyper_arg.add_argument('--batch_size', type=int, default=16, help="Number of point clouds read by iteration in each device for training.")
hyper_arg.add_argument('--val_batch_size', type=int, default=8, help="Number of point clouds read by iteration in each device for validation.")
hyper_arg.add_argument('--test_batch_size', type=int, default=8, help="Number of point clouds read by iteration in each device for testing.")
hyper_arg.add_argument('--num_workers', type=int, default=4, help='num workers for train dataloader')
hyper_arg.add_argument('--num_val_workers', type=int, default=2, help='num workers for val/test dataloader')
hyper_arg.add_argument('--ignore_label', type=str2list, default=255, help='index of a label to be ignored by the Loss function during training. Default to 255 - not ignoring.')

hyper_arg.add_argument('--pc_type', type=str, default='qRankMask_5',
                      help='the point cloud type to be used, which correspond to the desired quantile rank contour used to create the representation.',
                      choices=['qRankMask', 'qRank0.5', 'qRank0.7', 'qRank0.75', 'qRank0.8',
                               'qRank0.85', 'qRank0.9', 'qRank0.95', 'qRankMask_5'])
hyper_arg.add_argument('--kfold', type=int, default=13,
                      help='the k-fold group to be used for testing and validation. '
                           'The other groups will be used for training.')
hyper_arg.add_argument(
    '--rotation_rate',
    type=float,
    default=0.5,
    help='Percentage of random rotation rate R to be applied to the train dataset during training (augmentation) - real time oversampling. '
         'It random rotates R percent of the training input point cloud in the 3 axis')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='set to False for testing')
train_arg.add_argument('--log_freq', type=int, default=3000, help='statistics logging frequency in number of training steps for the Trainer setup - checkpoint save frequency in terms of steps')
train_arg.add_argument('--val_freq', type=int, default=1500, help='validation frequency in number of steps')
train_arg.add_argument(
    '--empty_cache_freq', type=int, default=1, help='Clear pytorch cache frequency in number of steps')

train_arg.add_argument(
        '--stochastic_weight_avg',
    type=str2bool,
    default=False,
    help='Apply the Stochastic Weight Averaging (SWA), which can make your DL models generalize better at virtually no additional cost. This can be used with both non-trained and trained DL models. The SWA procedure smooths the loss landscape thus making it harder to end up in a local minimum during optimization.')

train_arg.add_argument(
    '--loss_func',
    type=str,
    default='SL',
    help='Selects the desired loss function: Cross Entropy Loss (CE) or Symmetric Cross entropy Learning (SL)',
    choices = ['CE', 'SL']
)
train_arg.add_argument(
    '--loss_weights',
    type=str2flist,
    default='1',
    help='Set the class weights in the Loss function equals to the provided list, separated by comma. For no weights, '
         'set as 1 (default). It should follow the order of the vocabulary classes or mapping.')
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



# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--save_prediction', type=str2bool, default=False,
                      help="Boolean to save the point cloud predictions of the input tests.")
test_arg.add_argument('--save_pred_dir', type=str, default='outputs/pred',
                      help="The directory to store the point cloud predictions when save_prediction is True")


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True, help="Defines to use GPU (if cuda is available) or CPU.")
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'], help="Logging level.")
misc_arg.add_argument('--num_devices', type=int, default=1, help="Number of CPU or GPU devices to be used, depending on the is_cuda value. "
                                                             "If > 1 multi-CPU/GPU is enabled when possible, "
                                                             "else single CPU/GPU is used.")
misc_arg.add_argument('--gpu_index', type=str2list, default=None, help="the GPU index to be used when is_cuda is True. "
                                                                       "The indexes for multi-GPU must be comma separated. "
                                                                       "When informed, this parameter overwrites the num_devices value, which will be equal to the number of provided indexes. "
                                                                       "Otherwise the GPU indexes for the provided num_devices are automatically selected in numerical order.")
misc_arg.add_argument('--seed', type=int, default=123, help="Seed used for numpy random number generator to select the inputs index to read.")


def get_config():
    config = parser.parse_args()

    config.log_dir = config.log_dir + "_" + ('train' if config.is_train else 'test') + "_" + config.pc_type + "_kfold_" + \
                     str(config.kfold) + "_model-" + config.model + "_" + \
                     str(ctime().replace("  ", "_").replace(" ", "_").replace(":", "-"))
    return config  # Training settings
