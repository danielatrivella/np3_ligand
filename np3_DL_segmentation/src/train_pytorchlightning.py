# Code modified from https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/multigpu_lightning.py
#
# Paper "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755)
try:
    from pytorch_lightning.core import LightningModule
    from pytorch_lightning import Trainer
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install open3d pytorch_lightning`."
    )

import torch
import torch.nn as nn
import numpy as np
from src.SCELoss import SCELoss

from src.utils import get_prediction, WeightAverageMeter, ScoreAccuracy, HistIoU, save_predictions, fast_hist
from src.lig_pc_data_loader import ligands_dataloader
import MinkowskiEngine as ME

import logging, sys
from src.solvers import initialize_optimizer, initialize_scheduler

import pandas as pd
# torch.multiprocessing.set_start_method('spawn')

class MinkowskiSegmentationModule(LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine.
    """
    #
    def __init__(
        self,
        config,
        model,
        tb_logger,
        # optimizer_name="SGD",
        # lr=1e-3,
        # weight_decay=1e-5,
        # voxel_size=0.05,
        # batch_size=12,
        # val_batch_size=6,
        # train_num_workers=4,
        # val_num_workers=2,
    ):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        # save the config and any extra arguments
        self.save_hyperparameters(config)
        # read save the classes names
        self.class_names = config.class_names
        # initialize loss weights, 1 is no weight else use the provided list (splitted by ,)
        if config.loss_weights == [1]:
            self.loss_weights = torch.Tensor(np.asarray([1] * len(self.class_names)))
        else:
            self.loss_weights = torch.Tensor(config.loss_weights)
        # initialize loss function
        if config.loss_func == "CE":
            self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label,
                                                 weight=self.loss_weights)
        else:  # config.loss_func == "SL"
            # self.loss_weights = torch.Tensor([1, 3, 3.7])
            #print("\n\ndevice: ",self.device) before device was always 0
            self.criterion = SCELoss(self.device, alpha=config.SL_alpha,
                                beta=config.SL_beta,
                                num_classes=len(self.class_names),
                                weight=self.loss_weights)
        # print the loss func weights by class
        self.print_weights()
        # initialize the metrics
        self.loss_train = WeightAverageMeter()
        self.score_train = ScoreAccuracy()
        self.hist_IoU_train = HistIoU(self.class_names)
        self.loss_val = WeightAverageMeter()
        self.score_val = ScoreAccuracy()
        self.hist_IoU_val = HistIoU(self.class_names)
        self.loss_test = WeightAverageMeter()
        self.score_test = ScoreAccuracy()
        self.hist_IoU_test = HistIoU(self.class_names)
        # set test dataset as None
        self.test_dataset = None
        # log hyper params
        if self.tb_logger is not None:
            self.tb_logger.log_hyperparams(self.hparams,
                                        {"hyper_params/num_gpu": config.num_gpu,
                                         "hyper_params/total_bach_size":
                                             (config.num_gpu if config.num_gpu > 0 else 1) * config.batch_size * config.iter_size,
                                         "hyper_params/batch_size": config.batch_size,
                                         "hyper_params/iter_size": config.iter_size,
                                         "hyper_params/weight_decay": config.weight_decay})
            if config.optimizer == 'SGD':
                self.tb_logger.log_hyperparams(self.hparams,
                                            {"hyper_params/sgd_momentum": config.sgd_momentum,
                                             "hyper_params/sgd_dampening": config.sgd_dampening})
            elif config.optimizer == 'Adam':
                self.tb_logger.log_hyperparams(self.hparams,
                                            {"hyper_params/adam_beta1": config.adam_beta1,
                                             "hyper_params/adam_beta2": config.adam_beta2})
    # print the loss weights used for training
    def print_weights(self):
        class_names = self.class_names[self.loss_weights > 0.0]
        weights = self.loss_weights[self.loss_weights > 0.0]
        debug_str = "===> Training loss weights\n\nClasses:\t " + " ".join(class_names) + '\n'
        debug_str += 'Loss weights:\t ' + ' '.join('{:.03f}'.format(i) for i in weights) + '\n'
        logging.info(debug_str)
    # print metric formatted
    def print_metric_perclass(self, metric_values_perclass, data_type='Training', metric_type='IoU'):
        debug_str = "===> "+data_type+" "+metric_type+"\n\tClasses: " + " ".join(self.class_names) + '\n\t'
        debug_str += metric_type+':  \t' + ' '.join('{:.03f}'.format(i) for i in metric_values_perclass) + '\n'
        logging.info(debug_str)
    #
    def log_metrics_train(self, output, on_step=False, on_epoch=False):
        #print("global step",self.global_step)
        # update accuracy and loss
        #print('log metrics func')
        self.score_train(output['preds'], output['target'])
        self.loss_train(output['loss'], output['target'].size(0))
        self.hist_IoU_train(output['preds'], output['target'])
        #print("log metrics updated ")
        # log scalars
        self.log('train/train_loss', self.loss_train, on_step=on_step, on_epoch=on_epoch, sync_dist=on_epoch)
        self.log('train/train_acc', self.score_train, on_step=on_step, on_epoch=on_epoch, sync_dist=on_epoch)
        self.log('train/train_mIoU', self.hist_IoU_train, on_step=on_step, on_epoch=on_epoch, prog_bar=True, sync_dist=on_epoch)
        lrs = ', '.join(['{:.3e}'.format(x) for x in self.lr_schedulers().get_last_lr()])
        self.log('train/train_lr', float(lrs), on_step=on_step, on_epoch=on_epoch, prog_bar=True, sync_dist=on_epoch)
        # at the end of an epoch also log the IoU by class and print to file current progress
        if on_epoch:
            # log to file
            debug_str = "===> Epoch[{}/{}] Training: Loss {:.4f} - LR: {} - ".format(
                self.current_epoch+1, self.config.max_epoch, self.loss_train.compute().detach().item(), lrs)
            debug_str += "Score {:.3f}, mIoU {:.3f}".format(
                self.score_train.compute().detach().item(), self.hist_IoU_train.compute().detach().item())
            # log f1 score and precision and recall
            f1_per_class = self.hist_IoU_train.compute_f1_dice()
            precision_per_class = self.hist_IoU_train.compute_precision()
            recall_per_class = self.hist_IoU_train.compute_recall()
            debug_str += " - Macro - F1-Dice {:.3f}, Precision {:.3f}, Recall {:.3f}".format(
                f1_per_class.mean(), precision_per_class.mean(), recall_per_class.mean())
            logging.info(debug_str)
            # compute iou per class
            iou_per_class = self.hist_IoU_train.compute_iou()
            self.print_metric_perclass(iou_per_class)
            # log IoU to tensorboard
            for i in range(len(iou_per_class)):
                self.log('train_IoU/'+self.class_names[i],
                         iou_per_class[i], on_step=on_step, on_epoch=on_epoch, sync_dist=True)
            # log f1 score and precision and recall
            self.print_metric_perclass(f1_per_class, data_type='Train', metric_type="F1")
            self.print_metric_perclass(precision_per_class, data_type='Train', metric_type="Precision")
            self.print_metric_perclass(recall_per_class, data_type='Train', metric_type="Recall")
    #
    def log_metrics_val(self, output, on_step=False, on_epoch=False):
        # update accuracy and loss, on_epoch for val is always true
        # print('log metrics')
        self.score_val(output['preds'], output['target'])
        self.loss_val(output['loss'], output['target'].size(0))
        self.hist_IoU_val(output['preds'], output['target'])
        # log scalars
        self.log('validation/val_loss', self.loss_val, on_step=on_step, on_epoch=True, sync_dist=on_epoch)
        self.log('validation/val_acc', self.score_val, on_step=on_step, on_epoch=True, sync_dist=on_epoch)
        self.log('validation/val_mIoU', self.hist_IoU_val, on_step=on_step, on_epoch=True, prog_bar=True, sync_dist=on_epoch)
        # at the end of an epoch also log the IoU by class and print to file current progress
        if on_epoch:
            lrs = ', '.join(['{:.3e}'.format(x) for x in self.lr_schedulers().get_last_lr()])
            # log to file
            debug_str = "===> Epoch[{}/{}] Validation: Loss {:.4f} - LR: {} - ".format(
                self.current_epoch+1, self.config.max_epoch, self.loss_val.compute().detach().item(), lrs)
            debug_str += "Score {:.3f}, mIoU {:.3f}".format(
                self.score_val.compute().detach().item(), self.hist_IoU_val.compute().detach().item())
            # log f1 score and precision and recall
            f1_per_class = self.hist_IoU_val.compute_f1_dice()
            precision_per_class = self.hist_IoU_val.compute_precision()
            recall_per_class = self.hist_IoU_val.compute_recall()
            debug_str += " - Macro - F1-Dice {:.3f}, Precision {:.3f}, Recall {:.3f}".format(
                f1_per_class.mean(), precision_per_class.mean(), recall_per_class.mean())
            logging.info(debug_str)
            # compute iou per class
            iou_per_class = self.hist_IoU_val.compute_iou()
            self.print_metric_perclass(iou_per_class, data_type='Validation')
            # log IoU to tensorboard
            for i in range(len(iou_per_class)):
                self.log('validation_IoU/'+self.class_names[i],
                         iou_per_class[i], on_step=on_step, on_epoch=True, sync_dist=True) # sync dist across checkpoints, in not tensor metrics
            # log f1 score and precision and recall
            self.print_metric_perclass(f1_per_class, data_type='Validation', metric_type="F1")
            self.print_metric_perclass(precision_per_class, data_type='Validation', metric_type="Precision")
            self.print_metric_perclass(recall_per_class, data_type='Validation', metric_type="Recall")

    #
    def log_metrics_test(self, output, on_step=False, on_epoch=False):
        # update accuracy and loss, on_epoch for test is always true
        #print('log metrics func')
        self.score_test(output['preds'], output['target'])
        self.loss_test(output['loss'], output['target'].size(0))
        self.hist_IoU_test(output['preds'], output['target'])
        #print("updated metrics")
        # log scalars
        self.log('test/test_loss', self.loss_test, on_step=on_step, on_epoch=True, sync_dist=on_epoch)
        self.log('test/test_acc', self.score_test, on_step=on_step, on_epoch=True, sync_dist=on_epoch)
        self.log('test/test_mIoU', self.hist_IoU_test, on_step=on_step, on_epoch=True, sync_dist=on_epoch)
        # at the end of an epoch also log the IoU by class and print to file current progress
        if on_epoch:
            lrs = self.config.lr
            # log to file
            debug_str = "===> Epoch[{}/{}] Test: Loss {:.4f} - LR: {} - ".format(
                self.current_epoch+1, self.config.max_epoch, self.loss_test.compute().detach().item(), lrs)
            debug_str += "Score {:.3f}, mIoU {:.3f}".format(
                self.score_test.compute().detach().item(), self.hist_IoU_test.compute().detach().item())
            # compute f1 score and precision and recall
            f1_per_class = self.hist_IoU_test.compute_f1_dice()
            precision_per_class = self.hist_IoU_test.compute_precision()
            recall_per_class = self.hist_IoU_test.compute_recall()
            debug_str += " - Macro - F1-Dice {:.3f}, Precision {:.3f}, Recall {:.3f}".format(
                f1_per_class.mean(), precision_per_class.mean(), recall_per_class.mean())
            logging.info(debug_str)
            # compute iou per class
            iou_per_class = self.hist_IoU_test.compute_iou()
            self.print_metric_perclass(iou_per_class, data_type='Test')
            # log IoU to tensorboard
            for i in range(len(iou_per_class)):
                self.log('test_IoU/'+self.class_names[i],
                         iou_per_class[i], on_step=on_step, on_epoch=True, sync_dist=True)
            # log f1 score and precision and recall
            self.print_metric_perclass(f1_per_class, data_type='Test', metric_type="F1")
            self.print_metric_perclass(precision_per_class, data_type='Test', metric_type="Precision")
            self.print_metric_perclass(recall_per_class, data_type='Test', metric_type="Recall")

        elif self.config.save_prediction and self.config.test_batch_size == 1:
            # at the end of each step save the predictions
            # the quantitazion divided the coords by the grid_space in the reading,
            # # revert this operation before saving by multiplying by the grid_space
            save_predictions(output['coords'].numpy()*self.test_dataset.get_grid_space(),
                             output['target'], output['preds'],
                             self.test_dataset.get_entry_id(output['batch_idx']),
                             fast_hist(output['preds'], output['target'], len(self.class_names)),
                             self.config.save_pred_dir)
    #
    def train_dataloader(self):
        logging.info('\n===> Initialize train_dataloader')
        ligs_dataloader = ligands_dataloader(
            self.config,
            'train')
        if len(ligs_dataloader.dataset.vocab) != len(self.class_names) or (ligs_dataloader.dataset.vocab != self.class_names).any():
            print('ERROR incompatibility between the dataset vocab and the trainer class names')
            print('Dataset vocab = ', ligs_dataloader.dataset.vocab)
            print('Trainer class names = ', self.class_names)
            sys.exit('ERROR incompatibility between the train dataset vocabulary and the trainer class names')
        return ligs_dataloader
    #
    def val_dataloader(self):
        logging.info('\n===> Initialize val_dataloader')
        return ligands_dataloader(
            self.config,
            'val')
    #
    def test_dataloader(self):
        logging.info('\n===> Initialize test_dataloader')
        test_dl = ligands_dataloader(
            self.config,
            'test')
        if self.config.save_prediction and self.config.test_batch_size == 1:
            # save the dataset when saving the predicitons - to access the current ligID
            self.test_dataset = test_dl.dataset
        return test_dl
    #
    def forward(self, x):
        return self.model(x)
    #
    def training_step(self, batch, batch_idx):
        # print('start train')
        stensor = ME.SparseTensor(
            coordinates=batch[0], features=batch[1]
        )
        # Must clear cache at regular interval
        if self.global_step % self.config.empty_cache_freq == 0:
            torch.cuda.empty_cache()
        # backwards
        outs = self(stensor).F
        loss = self.criterion(outs, batch[2].long())
        preds = get_prediction(outs)
        return {'loss': loss, 'preds': preds.detach().cpu(), 'target': batch[2].long().detach().cpu()}
    #
    def training_step_end(self, outputs):
        if self.global_step % self.config.log_freq == 0 or self.global_step == 0:
            #print("log metrics ")
            self.log_metrics_train(outputs, on_step=True)
            #print("return log")
        return outputs
    #
    def training_epoch_end(self, outputs):
        #print("training_epoch_end current epoch", self.current_epoch)
        #print(outputs)
        if len(outputs) == 1:
            outs = outputs[0]
        else:
            # concat results
            outs = {}
            outs['loss'] = sum([out['loss'].detach().cpu() for out in outputs])/len(outputs)
            outs['preds'] = torch.cat([out['preds'] for out in outputs])
            outs['target'] = torch.cat([out['target'] for out in outputs])
        # print(outs)
        self.log_metrics_train(outs, on_epoch=True)
        # reset histogram for next epoch IoU computation, except in the last epoch
        #print("current_epoch", self.current_epoch)
        if self.current_epoch+1 != self.config.max_epoch:
            self.hist_IoU_train.reset()
        else:
            # save final confusion matrix in the last epoch
            confusion_m = self.hist_IoU_train.compute_confusion_matrix()
            m_not_nan = (np.diag(confusion_m) == np.diag(confusion_m))
            pd.DataFrame(confusion_m[m_not_nan][:, m_not_nan], columns=self.class_names[m_not_nan],
                         index=self.class_names[m_not_nan]).to_csv(self.config.log_dir + "/train_confusion_matrix.csv")
        #  single scheduler step, not possible when accumulating grad
        # sch = self.lr_schedulers()
        # sch.step()
    #
    def validation_step(self, batch, batch_idx):
        # print('start val')
        stensor = ME.SparseTensor(
            coordinates=batch[0], features=batch[1]
        )
        outs = self(stensor).F
        loss = self.criterion(outs, batch[2].long())
        preds = get_prediction(outs)
        # Must clear cache at regular interval
        if self.global_step % self.config.empty_cache_freq == 0:
            torch.cuda.empty_cache()
        return {'loss': loss, 'preds': preds.detach().cpu(), 'target': batch[2].long().detach().cpu()}
    #
    def validation_step_end(self, outputs):
        # update and log
        #print("validation step end")
        #print(outputs)
        if self.config.num_gpu > 1 and isinstance(outputs, list):
            # concat results
            outs = {}
            outs['loss'] = sum([out['loss'].detach().cpu() for out in outputs])/len(outputs)
            outs['preds'] = torch.cat([out['preds'] for out in outputs])
            outs['target'] = torch.cat([out['target'] for out in outputs])
        else:
            outs = outputs
        # log metrics
        if self.global_step % self.config.val_freq == 0 or self.global_step == 0:
            self.log_metrics_val(outs, on_epoch=False)
        return outs
    #
    def validation_epoch_end(self, outputs):
        #print("validation_epoch_end")
        #print(outputs)
        if len(outputs) == 1:
            outs = outputs[0]
        else:
            # concat results
            outs = {}
            outs['loss'] = sum([out['loss'].detach().cpu() for out in outputs])/len(outputs)
            outs['preds'] = torch.cat([out['preds'] for out in outputs])
            outs['target'] = torch.cat([out['target'] for out in outputs])
        # print(outs)
        self.log_metrics_val(outs, on_epoch=True)
    #
    def on_validation_end(self):
        # if last epoch, save validation confusion matrix
        if self.current_epoch + 1 == self.config.max_epoch:
            # save final confusion matrix
            confusion_m = self.hist_IoU_val.compute_confusion_matrix()
            m_not_nan = (np.diag(confusion_m) == np.diag(confusion_m))
            pd.DataFrame(confusion_m[m_not_nan][:, m_not_nan], columns=self.class_names[m_not_nan],
                         index=self.class_names[m_not_nan]).to_csv(self.config.log_dir + "/val_confusion_matrix.csv")
        # reset histogram for next epoch validation
        self.loss_val.reset()
        self.score_val.reset()
        self.hist_IoU_val.reset()
    #
    def on_test_start(self):
        # save prediction only if batch size equals 1, create the output directory and table
        if self.config.save_prediction and self.config.test_batch_size == 1 and self.config.num_gpu <= 1:
            import os
            save_pred_dir = self.config.save_pred_dir
            os.makedirs(save_pred_dir, exist_ok=True)
            logging.info('===> Saving predictions in: ' + save_pred_dir)
            if os.listdir(save_pred_dir):
                raise ValueError(f'Directory {save_pred_dir} not empty. '
                                 'Please remove the existing prediction.')
            else:
                # save header of ious by ligand entry
                # header: ligID, dataset_classnames, dataset_classnames confusion matrix cells
                with open(save_pred_dir + '/entries_ious.csv', "ab") as f:
                    np.savetxt(f, np.concatenate([['ligID'],
                                                  self.class_names,
                                                  [i + '_' + j for i in self.class_names for j in self.class_names
                                                   if i != j]
                                                  ]).reshape(
                        [1, 1 + len(self.class_names) * len(self.class_names)]),
                               delimiter=',', fmt='%s')
        elif self.config.save_prediction:
            logging.info('ERROR Saving predictions in: ' + self.config.save_pred_dir +
                         '\n----> The total allowed batch size must be equals 1')
            raise ValueError(f'When testing and saving the predictions the total allowed batch size must be equals 1.')
    #
    def test_step(self, batch, batch_idx):
        # print('start test')
        stensor = ME.SparseTensor(
            coordinates=batch[0], features=batch[1]
        )
        outs = self(stensor).F
        loss = self.criterion(outs, batch[2].long())
        preds = get_prediction(outs)
        # Must clear cache at regular interval
        if self.global_step % self.config.empty_cache_freq == 0:
            torch.cuda.empty_cache()
        if self.config.save_prediction and self.config.test_batch_size == 1:
            # also return the coords when sabing the predictions
            return {'loss': loss, 'preds': preds.detach().cpu(), 'target': batch[2].long().detach().cpu(),
                    'coords': batch[0].detach().cpu(), 'batch_idx': batch_idx}
        else:
            return {'loss': loss, 'preds': preds.detach().cpu(), 'target': batch[2].long().detach().cpu()}
    #
    def test_step_end(self, outputs):
        # update and log
        if self.config.num_gpu > 1 and isinstance(outputs, list):
            # concat results
            outs = {}
            outs['loss'] = sum([out['loss'].detach().cpu() for out in outputs])/len(outputs)
            outs['preds'] = torch.cat([out['preds'] for out in outputs])
            outs['target'] = torch.cat([out['target'] for out in outputs])
            if self.config.save_prediction and self.config.test_batch_size == 1:
                outs['coords'] = torch.cat([out['coords'] for out in outputs])
                outs['batch_idx'] = outputs['batch_idx']
        else:
            outs = outputs
        # log metrics
        if self.global_step % self.config.val_freq == 0 or self.global_step == 0:
            self.log_metrics_test(outs, on_epoch=False)
        return outs
    #
    def test_epoch_end(self, outputs):
        if len(outputs) == 1:
            outs = outputs[0]
        else:
            # concat results
            outs = {}
            outs['loss'] = sum([out['loss'].detach().cpu() for out in outputs])/len(outputs)
            outs['preds'] = torch.cat([out['preds'] for out in outputs])
            outs['target'] = torch.cat([out['target'] for out in outputs])
        # print(outs)
        self.log_metrics_test(outs, on_epoch=True)
    #
    def on_test_end(self):
        # save final confusion matrix
        confusion_m = self.hist_IoU_test.compute_confusion_matrix()
        m_not_nan = (np.diag(confusion_m) == np.diag(confusion_m))
        pd.DataFrame(confusion_m[m_not_nan][:, m_not_nan], columns=self.class_names[m_not_nan],
                     index=self.class_names[m_not_nan]).to_csv(self.config.log_dir + "/test_confusion_matrix.csv")
        # reset histogram for next test
        self.loss_test.reset()
        self.score_test.reset()
        self.hist_IoU_test.reset()
    #
    def predict_step(self, batch, batch_idx):
        # TensorField? https://github.com/NVIDIA/MinkowskiEngine/issues/367
        # print('start test')
        stensor = ME.SparseTensor(
            coordinates=batch[0], features=batch[1]
        )
        outs = self(stensor).F
        preds = get_prediction(outs)
        # Must clear cache at regular interval
        if self.global_step % self.config.empty_cache_freq == 0:
            torch.cuda.empty_cache()
        # return preds.detach().cpu()
        return {'preds': preds.detach().cpu(), 'coords': batch[0].detach().cpu(), 'batch_idx': batch_idx}
        # if self.config.save_prediction and self.config.test_batch_size == 1:
        #     # also return the coords when sabing the predictions
        #     return {'loss': loss, 'preds': preds.detach().cpu(), 'target': batch[2].long().detach().cpu(),
        #             'coords': batch[0].detach().cpu(), 'batch_idx': batch_idx}
        # else:
        #     return {'loss': loss, 'preds': preds.detach().cpu(), 'target': batch[2].long().detach().cpu()}
    #
    def configure_optimizers(self):
        optimizer = initialize_optimizer(self.model.parameters(), self.config)
        scheduler = initialize_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]

