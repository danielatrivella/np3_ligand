# code from https://github.com/chrischoy/SpatioTemporalSegmentation/
import json
import logging
import os
import errno
import time

import numpy as np
import torch
import open3d as o3d
# from lib.pc_utils import colorize_pointcloud, save_point_cloud

from torchmetrics import Metric

def available_cuda_memory_in_mbytes(safety_margin_for_gpu_memory=512):

    cuda_total_memory = torch.cuda.get_device_properties(torch.cuda.current_device).total_memory
    cuda_cached_memory = torch.cuda.memory_reserved(torch.cuda.current_device)

    available_cuda_memory_for_dbscan_in_mbytes = (
        cuda_total_memory - cuda_cached_memory
    ) / 1024 ** 2 - safety_margin_for_gpu_memory

    return available_cuda_memory_for_dbscan_in_mbytes


# random rotate 3d points using the open3d library
def random_rotate_3dpoints(points):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.asarray([2*np.pi*np.random.rand(),
                                                              2*np.pi*np.random.rand(),
                                                              2*np.pi*np.random.rand()]))
    pcd = pcd.rotate(R, center=pcd.get_center())
    return np.ascontiguousarray(pcd.points)

def colorize_pointcloud(xyz, label, ignore_label=255):
    assert label[label != ignore_label].max() < len(COLOR_MAP_RGB), 'Not enough colors.'
    label_rgb = np.array([COLOR_MAP_RGB[i] if i != ignore_label else IGNORE_COLOR for i in label])
    return np.hstack((xyz, label_rgb))


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
    """Save an RGB point cloud as a PLY file.
  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
    assert points_3d.ndim == 2
    if with_label:
        assert points_3d.shape[1] == 7
        python_types = (float, float, float, int, int, int, int)
        npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                     ('blue', 'u1'), ('label', 'u1')]
    else:
        if points_3d.shape[1] == 3:
            gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
            points_3d = np.hstack((points_3d, gray_concat))
        assert points_3d.shape[1] == 6
        python_types = (float, float, float, int, int, int)
        npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                     ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # Write
        PlyData([el]).write(filename)
    else:
        # PlyData([el], text=True).write(filename)
        with open(filename, 'w') as f:
            f.write('ply\n'
                    'format ascii 1.0\n'
                    'element vertex %d\n'
                    'property float x\n'
                    'property float y\n'
                    'property float z\n'
                    'property uchar red\n'
                    'property uchar green\n'
                    'property uchar blue\n'
                    'property uchar alpha\n'
                    'end_header\n' % points_3d.shape[0])
            for row_idx in range(points_3d.shape[0]):
                X, Y, Z, R, G, B = points_3d[row_idx]
                f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
    if verbose is True:
        print('Saved point cloud to: %s' % filename)


def load_state_with_same_shape(model, weights):
    model_state = model.state_dict()
    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
    return filtered_weights


def checkpoint(model, optimizer, epoch, iteration, config, best_val=None, best_val_iter=None, postfix=None):
    logging.info(f"Starting checkpoint")
    mkdir_p(config.log_dir)
    if config.overwrite_weights:
        if postfix is not None:
            filename = f"checkpoint_{config.wrapper_type}{config.model}{postfix}.pth"
        else:
            filename = f"checkpoint_{config.wrapper_type}{config.model}.pth"
    else:
        filename = f"checkpoint_{config.wrapper_type}{config.model}_iter_{iteration}.pth"
    checkpoint_file = config.log_dir + '/' + filename
    state = {
        'iteration': iteration,
        'epoch': epoch,
        'arch': config.model,
        'state_dict': (model.module.state_dict() if hasattr(model, 'module') else model.state_dict()), # check if model is wrapped in a torch.nn.parallel.DistributedDataParallel object
        'optimizer': optimizer.state_dict()
    }
    if best_val is not None:
        state['best_val'] = best_val
        state['best_val_iter'] = best_val_iter
    json.dump(vars(config), open(config.log_dir + '/config.json', 'w'), indent=4)
    torch.save(state, checkpoint_file)
    logging.info(f"Checkpoint saved to {checkpoint_file}")
    # Delete symlink if it exists
    if os.path.exists(f'{config.log_dir}/weights.pth'):
        os.remove(f'{config.log_dir}/weights.pth')
    # Create symlink
    os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')


def precision_at_one(pred, target, ignore_label=255):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != ignore_label]
    correct = correct.view(-1)
    if correct.nelement():
        return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
    else:
        return float('nan')


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k] + pred[k], minlength=n ** 2).reshape(n, n)

def fast_hist_torch(pred, label, n):
    k = (label >= 0) & (label < n)
    return torch.bincount(n * label[k] + pred[k], minlength=n ** 2).reshape(n, n)


# IOU = TP / (TP+FP+FN) = diag / (linha + coluna - diag)
def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

# ref: https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
# F1 = 2TP / (2TP+FP+FN) = 2*diag / (linha + coluna)
# macro-F1 = average F1 by class
# micro-F1 = micro-precision = micro-recall = accuracy
def per_class_dice_f1(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2*np.diag(hist) / (hist.sum(1) + hist.sum(0))

# precision = the number of correctly predicted values in a class out of all predicted values of that class (column)
# precision = diag / column
def per_class_precision(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / hist.sum(1)

# recall = the number of correctly predicted values in a class out of the number of actual values of that class (row)
# precision = diag / row
def per_class_recall(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / hist.sum(0)

# confusion matrix normalized by the row (target), where the total is the union TT FN FP by target
# TODO? could be normilzed by prediction (column) or by the total or None
def per_class_confusion_matrix(hist):
    confusion_matrix = np.zeros(hist.shape)
    # return hist
    with np.errstate(divide='ignore', invalid='ignore'):
        # total equals the union of the target TT, FN (row) plus the predictions FP (column)
        total_class = (hist.sum(1) + hist.sum(0) - np.diag(hist))
        for i in range(hist.shape[0]):
            # normalized by the target (rows)
            confusion_matrix[i, :] = hist[i, :] / total_class[i]
    return confusion_matrix


class WithTimer(object):
    """Timer for with statement."""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        out_str = 'Elapsed: %s' % (time.time() - self.tstart)
        if self.name:
            logging.info('[{self.name}]')
        logging.info(out_str)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.average_time = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class ExpTimer(Timer):
    """ Exponential Moving Average Timer """

    def __init__(self, alpha=0.5):
        super(ExpTimer, self).__init__()
        self.alpha = alpha

    def toc(self):
        self.diff = time.time() - self.start_time
        self.average_time = self.alpha * self.diff + \
                            (1 - self.alpha) * self.average_time
        return self.average_time


class AverageMeter(object):
    """Computes and stores the weighted average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ScoreAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        current_correct = torch.sum(preds == target)
        self.correct += current_correct.to(self.correct.device)
        self.total += target.numel()
        return current_correct/target.numel()
    def compute(self):
        # print('compute score', self.correct.float() / self.total)
        return self.correct.float() / self.total
    def reset(self):
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)

class WeightAverageMeter(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.Tensor([0]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, value: float, n: int):
        self.total += value.detach().to(self.total.device) * n
        self.count += n
    def compute(self):
        # print('compute wavgmeter', self.total / self.count)
        return self.total / self.count
    def reset(self):
        self.count = torch.tensor(0)
        self.total = torch.Tensor([0])

class HistIoU(Metric):
    def __init__(self, class_names, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("hist", default=torch.Tensor(np.zeros((len(class_names), len(class_names)))),
                       dist_reduce_fx="sum")
        self.class_names = class_names
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        current_hist = torch.Tensor(fast_hist(preds, target,
                                 len(self.class_names))).to(self.hist.device)
        self.hist += current_hist
        # return per_class_iu(current_hist) * 100
    def compute(self):
        res = per_class_iu(self.hist.detach().cpu()) * 100
        # print('compute hist iou', torch.Tensor([np.nanmean(res)]))
        return torch.Tensor([np.nanmean(res)])
    def compute_iou(self):
        return per_class_iu(self.hist.detach().cpu()) * 100
    def compute_confusion_matrix(self):
        return per_class_confusion_matrix(self.hist.detach().cpu()) * 100
    def compute_f1_dice(self):
        return per_class_dice_f1(self.hist.detach().cpu()) * 100
    def compute_precision(self):
        return per_class_precision(self.hist.detach().cpu()) * 100
    def compute_recall(self):
        return per_class_recall(self.hist.detach().cpu()) * 100
    def reset(self):
        self.hist = torch.Tensor(np.zeros((len(self.class_names), len(self.class_names))))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_txt(path):
    """Read txt file into lines.
  """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def debug_on():
    import sys
    import pdb
    import functools
    import traceback

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper

    return decorator


def get_prediction(output,dim=1):
    return output.max(dim=dim)[1]


def count_parameters(model):
    # print('count_parameters_norequires_grad'+str(np.sum([p.numel() for p in model.parameters()])))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda, index=0):
    if is_cuda:
        return torch.device('cuda', index)
    else:
        return torch.device('cpu')


class HashTimeBatch(object):

    def __init__(self, prime=5279):
        self.prime = prime

    def __call__(self, time, batch):
        return self.hash(time, batch)

    def hash(self, time, batch):
        return self.prime * batch + time

    def dehash(self, key):
        time = key % self.prime
        batch = key / self.prime
        return time, batch


def save_rotation_pred(iteration, pred, dataset, save_pred_dir):
    """Save prediction results in original pointcloud scale."""
    decode_label_map = {}
    for k, v in dataset.label_map.items():
        decode_label_map[v] = k
    pred = np.array([decode_label_map[x] for x in pred], dtype=np.int64)
    out_rotation_txt = dataset.get_output_id(iteration) + '.txt'
    out_rotation_path = save_pred_dir + '/' + out_rotation_txt
    np.savetxt(out_rotation_path, pred, fmt='%i')

elements_color_SP_test = {'0': np.array([0, 0, 0]), '1': np.array([0, 1, 1]),
                          '2': np.array([1, 1, 0]), '3': np.array([167, 162, 132]) / 255,
                          '4': np.array([208,200,142]) / 255, '5': np.array([242,132,130]) / 255,
                          '6': np.array([255, 136, 17]) / 255, '7': np.array([147, 196, 125]) / 255,
                          '8': np.array([84,87,124]) / 255, '9': np.array([220, 127, 155]) / 255,
                          '10': np.array([150, 173, 200]) / 255, '11': np.array([167, 194, 193]) / 255,
                          '12': np.array([183, 214, 186]) / 255, '13': np.array([215, 255, 171]) / 255,
                          '14': np.array([234, 255, 140]) / 255, '15': np.array([252, 255, 108]) / 255,
                          '16': np.array([242,132,130]) / 255, '17': np.array([163, 113, 91]) / 255,
                          '18': np.array([109, 69, 76]) / 255, '19': np.array([122, 86, 92]) / 255,
                          '20': np.array([230,145,56]) / 255, '21': np.array([0,255,105]) / 255,
                          '22': np.array([0, 1, 1]), '23': np.array([0, 1, 0]),
                          '24': np.array([0, 0.5, 0])}

def save_predictions(coords, labels, pred, ligID, entry_hist, save_pred_dir):
    """Save predictions results in the voxelized coords scale, save original coords and labels and the predicted labels
    Save the ious by entry"""
    # coords = coords.numpy() # converting to numpy in the function call -> reverting the spacing by * grid_space
    labels = labels.numpy().astype(int)
    pred = pred.numpy().astype(int)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords[:,1:])
    # save correct labels and the predicted labels pc
    pcd.colors = o3d.utility.Vector3dVector(np.asarray([np.array([v, v, v]) for v in labels]))
    o3d.io.write_point_cloud(save_pred_dir+'/'+ligID+'_target.xyzrgb', pcd)
    # o3d.visualization.draw_geometries([pcd])
    pcd.colors = o3d.utility.Vector3dVector(np.asarray([np.array([v, v, v]) for v in pred]))
    o3d.io.write_point_cloud(save_pred_dir + '/' + ligID + '_predicted.xyzrgb', pcd)
    # o3d.visualization.draw_geometries([pcd])
    entry_confusion_matrix = per_class_confusion_matrix(entry_hist) * 100
    ious = np.diag(entry_confusion_matrix)
    with open(save_pred_dir + '/entries_ious.csv', "ab") as f:
        np.savetxt(f, np.concatenate([[ligID], ious,
                                      [entry_confusion_matrix[i,j] for i in range(entry_confusion_matrix.shape[0])
                                       for j in range(entry_confusion_matrix.shape[1]) if i != j]]).reshape(
            [1, 1 + len(ious)*len(ious)]),
                   delimiter=',', fmt='%s')

# def save_predictions(coords, upsampled_pred, transformation, dataset, config, iteration,
#                      save_pred_dir):
#     """Save prediction results in original pointcloud scale."""
#     from lib.dataset import OnlineVoxelizationDatasetBase
#     if dataset.IS_ONLINE_VOXELIZATION:
#         assert transformation is not None, 'Need transformation matrix.'
#     iter_size = coords[:, -1].max() + 1  # Normally batch_size, may be smaller at the end.
#     if dataset.IS_TEMPORAL:  # Iterate over temporal dilation.
#         iter_size *= config.temporal_numseq
#     for i in range(iter_size):
#         # Get current pointcloud filtering mask.
#         if dataset.IS_TEMPORAL:
#             j = i % config.temporal_numseq
#             i = i // config.temporal_numseq
#         batch_mask = coords[:, -1].numpy() == i
#         if dataset.IS_TEMPORAL:
#             batch_mask = np.logical_and(batch_mask, coords[:, -2].numpy() == j)
#         # Calculate original coordinates.
#         coords_original = coords[:, :3].numpy()[batch_mask] + 0.5
#         if dataset.IS_ONLINE_VOXELIZATION:
#             # Undo voxelizer transformation.
#             curr_transformation = transformation[i, :16].numpy().reshape(4, 4)
#             xyz = np.hstack((coords_original, np.ones((batch_mask.sum(), 1))))
#             orig_coords = (np.linalg.inv(curr_transformation) @ xyz.T).T
#         else:
#             orig_coords = coords_original
#         orig_pred = upsampled_pred[batch_mask]
#         # Undo ignore label masking to fit original dataset label.
#         if dataset.IGNORE_LABELS:
#             if isinstance(dataset, OnlineVoxelizationDatasetBase):
#                 label2masked = dataset.label2masked
#                 maskedmax = label2masked[label2masked < 255].max() + 1
#                 masked2label = [label2masked.tolist().index(i) for i in range(maskedmax)]
#                 orig_pred = np.take(masked2label, orig_pred)
#             else:
#                 decode_label_map = {}
#                 for k, v in dataset.label_map.items():
#                     decode_label_map[v] = k
#                 orig_pred = np.array([decode_label_map[x] for x in orig_pred], dtype=np.int64)
#         # Determine full path of the destination.
#         full_pred = np.hstack((orig_coords[:, :3], np.expand_dims(orig_pred, 1)))
#         filename = 'pred_%04d_%02d.npy' % (iteration, i)
#         if dataset.IS_TEMPORAL:
#             filename = 'pred_%04d_%02d_%02d.npy' % (iteration, i, j)
#         # Save final prediction as npy format.
#         np.save(os.path.join(save_pred_dir, filename), full_pred)


def visualize_results(coords, input, target, upsampled_pred, config, iteration):
    # Get filter for valid predictions in the first batch.
    target_batch = coords[:, 3].numpy() == 0
    input_xyz = coords[:, :3].numpy()
    target_valid = target.numpy() != 255
    target_pred = np.logical_and(target_batch, target_valid)
    target_nonpred = np.logical_and(target_batch, ~target_valid)
    ptc_nonpred = np.hstack((input_xyz[target_nonpred], np.zeros((np.sum(target_nonpred), 3))))
    # Unwrap file index if tested with rotation.
    file_iter = iteration
    if config.test_rotation >= 1:
        file_iter = iteration // config.test_rotation
    # Create directory to save visualization results.
    os.makedirs(config.visualize_path, exist_ok=True)
    # Label visualization in RGB.
    xyzlabel = colorize_pointcloud(input_xyz[target_pred], upsampled_pred[target_pred])
    xyzlabel = np.vstack((xyzlabel, ptc_nonpred))
    filename = '_'.join([config.dataset, config.model, 'pred', '%04d.ply' % file_iter])
    save_point_cloud(xyzlabel, os.path.join(config.visualize_path, filename), verbose=False)
    # RGB input values visualization.
    xyzrgb = np.hstack((input_xyz[target_batch], input[:, :3].cpu().numpy()[target_batch]))
    filename = '_'.join([config.dataset, config.model, 'rgb', '%04d.ply' % file_iter])
    save_point_cloud(xyzrgb, os.path.join(config.visualize_path, filename), verbose=False)
    # Ground-truth visualization in RGB.
    xyzgt = colorize_pointcloud(input_xyz[target_pred], target.numpy()[target_pred])
    xyzgt = np.vstack((xyzgt, ptc_nonpred))
    filename = '_'.join([config.dataset, config.model, 'gt', '%04d.ply' % file_iter])
    save_point_cloud(xyzgt, os.path.join(config.visualize_path, filename), verbose=False)


def permute_pointcloud(input_coords, pointcloud, transformation, label_map,
                       voxel_output, voxel_pred):
    """Get permutation from pointcloud to input voxel coords."""

    def _hash_coords(coords, coords_min, coords_dim):
        return np.ravel_multi_index((coords - coords_min).T, coords_dim)

    # Validate input.
    input_batch_size = input_coords[:, -1].max().item()
    pointcloud_batch_size = pointcloud[:, -1].max().int().item()
    transformation_batch_size = transformation[:, -1].max().int().item()
    assert input_batch_size == pointcloud_batch_size == transformation_batch_size
    pointcloud_permutation, pointcloud_target = [], []
    # Process each batch.
    for i in range(input_batch_size + 1):
        # Filter batch from the data.
        input_coords_mask_b = input_coords[:, -1] == i
        input_coords_b = (input_coords[input_coords_mask_b])[:, :-1].numpy()
        pointcloud_b = pointcloud[pointcloud[:, -1] == i, :-1].numpy()
        transformation_b = transformation[i, :-1].reshape(4, 4).numpy()
        # Transform original pointcloud to voxel space.
        original_coords1 = np.hstack((pointcloud_b[:, :3], np.ones((pointcloud_b.shape[0], 1))))
        original_vcoords = np.floor(original_coords1 @ transformation_b.T)[:, :3].astype(int)
        # Hash input and voxel coordinates to flat coordinate.
        vcoords_all = np.vstack((input_coords_b, original_vcoords))
        vcoords_min = vcoords_all.min(0)
        vcoords_dims = vcoords_all.max(0) - vcoords_all.min(0) + 1
        input_coords_key = _hash_coords(input_coords_b, vcoords_min, vcoords_dims)
        original_vcoords_key = _hash_coords(original_vcoords, vcoords_min, vcoords_dims)
        # Query voxel predictions from original pointcloud.
        key_to_idx = dict(zip(input_coords_key, range(len(input_coords_key))))
        pointcloud_permutation.append(
            np.array([key_to_idx.get(i, -1) for i in original_vcoords_key]))
        pointcloud_target.append(pointcloud_b[:, -1].astype(int))
    pointcloud_permutation = np.concatenate(pointcloud_permutation)
    # Prepare pointcloud permutation array.
    pointcloud_permutation = torch.from_numpy(pointcloud_permutation)
    permutation_mask = pointcloud_permutation >= 0
    permutation_valid = pointcloud_permutation[permutation_mask]
    # Permuate voxel output to pointcloud.
    pointcloud_output = torch.zeros(pointcloud.shape[0], voxel_output.shape[1]).to(voxel_output)
    pointcloud_output[permutation_mask] = voxel_output[permutation_valid]
    # Permuate voxel prediction to pointcloud.
    # NOTE: Invalid points (points found in pointcloud but not in the voxel) are mapped to 0.
    pointcloud_pred = torch.ones(pointcloud.shape[0]).int().to(voxel_pred) * 0
    pointcloud_pred[permutation_mask] = voxel_pred[permutation_valid]
    # Map pointcloud target to respect dataset IGNORE_LABELS
    pointcloud_target = torch.from_numpy(
        np.array([label_map[i] for i in np.concatenate(pointcloud_target)])).int()
    return pointcloud_output, pointcloud_pred, pointcloud_target

def draw_pc(coords, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(labels)
    o3d.visualization.draw_geometries([pcd])
