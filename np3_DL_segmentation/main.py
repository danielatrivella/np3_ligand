# main training and testing pipeline, call the pytorch lightning module
import logging
#import json
# from easydict import EasyDict as edict

# Torch packages
import torch
try:
    from pytorch_lightning.core import LightningModule
    from pytorch_lightning import Trainer
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install open3d pytorch_lightning`."
    )
from pytorch_lightning import loggers as pl_loggers

import numpy as np
import pandas as pd
import MinkowskiEngine as ME
from config import get_config
from src.utils import mkdir_p,load_state_with_same_shape,count_parameters
from src.load_models import load_model
from src.train_pytorchlightning import MinkowskiSegmentationModule
import sys, os
# Change dataloader multiprocess start method to anything not fork
# torch.multiprocessing.set_start_method('spawn')

def main():
    config = get_config()

    if config.is_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    if config.is_cuda and max(config.gpu_index) >= torch.cuda.device_count():
        raise Exception("Wrong GPU index ("+str(config.gpu_index)+"). The number of devices is "+
                        str(torch.cuda.device_count())+".")
    elif not config.is_cuda:
        # set fixed this parameters if is_cuda is False
        config.accelerator = 'cpu'
        config.gpu_index = None
        config.num_gpu = 0
    elif config.is_cuda:
        # set parameters for cuda True
        config.accelerator = 'gpu'
        if len(config.gpu_index) != config.num_gpu:
            raise Exception("Wrong number of GPU indexes ("+str(config.gpu_index)+"). It should match with the number of GPUs "+
                        str(config.num_gpu)+".")


    # Trainer configs
    config.strategy = 'ddp'

    # set the stdout logging level
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    # add logging file handler
    # create file handler to log the training progress
    mkdir_p(config.log_dir)
    fh = logging.FileHandler(config.log_dir + '/job_info.log')
    fh.setLevel(logging.INFO)
    # set basic config to both handlers
    logging.basicConfig(
        format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
        datefmt='%m/%d %H:%M:%S',
        handlers=[ch, fh])
    # add the handlers to the logger
    logging.getLogger().addHandler(fh)
    # create a logger for the model checkpoint and tensorboard info
    tb_logger = pl_loggers.TensorBoardLogger(config.log_dir, name='model', default_hp_metric=False)

    logging.info('===> Configurations')
    dconfig = vars(config)
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    # compute class names based on the mapping, if any
    # read vocab to extract the classes names
    class_names = np.asarray([line.rstrip('\n') for line in open(config.vocab_path)] + ["solvent_background"])
    if config.class_mapping_path:
        mapping = pd.read_csv(config.class_mapping_path)
        mapping = mapping[['mapping', 'target']][~mapping[['mapping', 'target']].duplicated()].sort_values('target')
        class_names = mapping.mapping.values
    # add the mapped class names to the config dict
    config.class_names = class_names

    ######
    logging.info('===> Building model')
    ######
    # use model argument to select the ME UNet model
    net = load_model(config.model)
    # initialize model
    # removed multiple in channels -> (3 if "qRankMask_" in config.pc_type else 1), all image types now use only one in channel
    model = net(in_channels=1, out_channels=len(class_names), config=config, D=3)
    if config.num_gpu > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    
    logging.info(model)
    logging.info("Number of trainable parameters: " + str(count_parameters(model)))
    # Load weights if specified by the parameter.
    if config.weights.lower() != 'none':
        logging.info('===> Loading weights: ' + config.weights)
        # configure map_location properly, setting the main device when using gpu
        # map_location = ({'cuda:%d' % config.gpu_index: 'cuda:%d' % gpu} if gpu is not None else device)
        map_location = 'cpu'
        state = torch.load(config.weights, map_location=map_location)
        if 'state_dict' not in state.keys():
            state = {'state_dict': state}
        #print(state['state_dict'])
        # remove model. prefix from the parameters names
        state['state_dict'] = {(k.partition('model.')[2] if k.startswith('model.') else k): state['state_dict'][k]
                               for k in state['state_dict'].keys() if k not in ['criterion.weight', 'model.criterion.weight', 'criterion.cross_entropy.weight', 'model.criterion.cross_entropy.weight']}
        logging.info(state['state_dict'])
        if config.weights_for_inner_model:
            model.load_state_dict(state['state_dict'])
        else:
            if config.lenient_weight_loading:
                matched_weights = load_state_with_same_shape(model, state['state_dict'])
                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['state_dict'])
        del state
    else:
        model.weight_initialization()

    # create the pytorch lightning module
    pl_module = MinkowskiSegmentationModule(config, model, tb_logger)
    ####
    # initialize trainer
    trainer = Trainer(max_epochs=config.max_epoch,
                      devices=config.gpu_index,
                      accelerator=config.accelerator,
                      strategy=config.strategy,
                      log_every_n_steps=config.log_freq,
                      logger=tb_logger,
                      accumulate_grad_batches=config.iter_size)
    
    # run training pipeline or only testing
    if config.is_train:
        # run model fit
        if config.resume:
            trainer.fit(pl_module,ckpt_path=config.resume)
        else:
            trainer.fit(pl_module)
        torch.cuda.empty_cache()  # empty cache before testing
        # test best model
        trainer.test(ckpt_path="best")
    else:
        # test using provided weights if any
        trainer.test(pl_module)



if __name__ == '__main__':
    __spec__ = None
    main()
