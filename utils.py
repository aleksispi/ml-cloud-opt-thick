import os
import numpy as np
import time
import random
from sklearn.metrics import jaccard_score
from collections import OrderedDict
from scipy.special import expit
import gc
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def replace(string_in, replace_from, replace_to='_'):
    if not isinstance(replace_from, list):
        replace_from = [replace_from]
    string_out = string_in
    for replace_entry in replace_from:
        string_out = string_out.replace(replace_entry, replace_to)
    return string_out


class BaseStat():
    """
    Basic statistic from which all other statistic types inherit
    """
    def __init__(self, name):
        self.name = name
        self.ep_idx = 0
        self.stat_collector = None

    def collect(self, value):
        pass

    def get_data(self):
        return {}

    def next_step(self):
        pass

    def next_ep(self):
        self.ep_idx += 1

    def next_batch(self):
        pass

    def compute_mean(self, mean, value, counter):
        return (counter * mean + value) / (counter + 1)

    def compute_ma(self, ma, value, ma_weight):
        return (1 - ma_weight) * ma + ma_weight * value


class AvgStat(BaseStat):
    """
    Standard average statistic (can track total means, moving averages,
    exponential moving averages etcetera)
    """
    def __init__(self, name, coll_freq='ep', ma_weight=0.001):
        super(AvgStat, self).__init__(name=name)
        self.counter = 0
        self.mean = 0.0
        self.ma = 0.0
        self.last = None
        self.means = []
        self.mas = []
        self.values = []
        self.times = []
        self.coll_freq = coll_freq
        self.ma_weight = ma_weight

    def collect(self, value, delta_counter=1):
        self.counter += delta_counter

        self.values.append(value)
        self.times.append(self.counter)
        self.mean = self.compute_mean(self.mean, value, len(self.means))
        self.means.append(self.mean)
        if self.counter < 10:
            # Want the ma to be more stable early on
            self.ma = self.mean
        else:
            self.ma = self.compute_ma(self.ma, value, self.ma_weight)
        self.mas.append(self.ma)
        self.last = value

    def get_data(self):
        return {'times': self.times, 'means': self.means, 'mas': self.mas, 'values': self.values}

    def print(self, timestamp=None):
        if self.counter <= 0:
            return
        self._print_helper()

    def _print_helper(self, mean=None, ma=None, last=None):

        # Set defaults
        if mean is None:
            mean = self.mean
        if ma is None:
            ma = self.ma
        if last is None:
            last = self.last

        if isinstance(mean, float):
            print('Mean %-35s tot: %10.5f, ma: %10.5f, last: %10.5f' %
                  (self.name, mean, ma, last))
        else:
            print('Mean %-35s tot:  (%.5f' % (self.name, mean[0]), end='')
            for i in range(1, mean.size - 1):
                print(', %.5f' % mean[i], end='')
            print(', %.5f)' % mean[-1])
            print('%-40s ma:   (%.5f' % ('', ma[0]), end='')
            for i in range(1, ma.size - 1):
                print(', %.5f' % ma[i], end='')
            print(', %.5f)' % ma[-1])
            print('%-40s last: (%.5f' % ('', last[0]), end='')
            for i in range(1, last.size - 1):
                print(', %.5f' % last[i], end='')
            print(', %.5f)' % last[-1])

    def save(self, save_dir):
        file_name = replace(self.name, [' ', '(', ')', '/'], '-')
        file_name = replace(file_name, ['<', '>'], '')
        file_name += '.npz'
        np.savez(os.path.join(save_dir, file_name),
                 values=np.asarray(self.values), means=np.asarray(self.means),
                 mas=np.asarray(self.mas), times=np.asarray(self.times))

    def plot(self, times=None, values=None, means=None, mas=None, save_dir=None):
        # Set defaults
        if times is None:
            times = self.times
        if values is None:
            values = self.values
        if means is None:
            means = self.means
        if mas is None:
            mas = self.mas
        if save_dir is None:
            save_dir_given = None
            save_dir = os.path.join(self.log_dir, 'stats', 'data')
        else:
            save_dir_given = save_dir

        # Define x-label
        if self.coll_freq == 'ep':
            xlabel = 'episode'
        elif self.coll_freq == 'step':
            xlabel = 'step'

        if np.asarray(values).ndim > 1:
            # Plot all values
            self._plot(times, values, self.name + ' all', xlabel, 'y', None,
                       save_dir_given)

            # Plot total means
            self._plot(times, means, self.name + ' total mean', xlabel, 'y', None,
                       save_dir_given)

            # Plot moving averages
            self._plot(times, mas, self.name + ' total exp ma', xlabel, 'y', None,
                       save_dir_given)
        else:
            self._plot_in_same(times, [values, means, mas],
                               self.name, xlabel, 'y',
                               ['all-data', 'mean', 'ma'],
                               [None, '-.', '-'], [0.25, 1.0, 1.0],
                               save_dir_given)

        # Also save current data to file
        if save_dir_given is None:
            file_name = replace(self.name, [' ', '(', ')', '/'], '-')
            file_name = replace(file_name, ['<', '>'], '')
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, file_name), values)

    def _plot(self, x, y, title='plot', xlabel='x', ylabel='y', legend=None,
              log_dir=None):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend is None:
            plt.legend([str(k) for k in range(np.asarray(y).shape[1])])
        else:
            plt.legend(legend)
        title_to_save = replace(title, [' ', '(', ')', '/'], '-')
        title_to_save = replace(title_to_save, ['<', '>'], '')
        if log_dir is None:
            log_dir = os.path.join(self.log_dir, 'stats', 'plots')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        plt.savefig(os.path.join(log_dir, title_to_save + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()

    def _plot_in_same(self, x, ys, title='plot', xlabel='x', ylabel='y',
                      legend=None, line_styles=None, alphas=None,
                      log_dir=None):
        if alphas is None:
            alphas = [1.0 for _ in range(len(ys))]
        plt.figure()
        for i in range(len(ys)):
            if line_styles[i] is not None:
                plt.plot(x, ys[i],
                         linestyle=line_styles[i], alpha=alphas[i])
            else:
                plt.plot(x, ys[i], 'yo', alpha=alphas[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend is None:
            plt.legend([str(k) for k in range(np.asarray(y).shape[1])])
        else:
            plt.legend(legend)
        title_to_save = replace(title, [' ', '(', ')', '/'], '-')
        title_to_save = replace(title_to_save, ['<', '>'], '')
        if log_dir is None:
            log_dir = os.path.join(self.log_dir, 'stats', 'plots')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        plt.savefig(os.path.join(log_dir, title_to_save + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()


class StatCollector():
    """
    Statistics collector class
    """
    def __init__(self, log_dir, tot_nbr_steps, print_iter):
        self.stats = OrderedDict()
        self.log_dir = log_dir
        self.ep_idx = 0
        self.step_idx = 0
        self.epoch_idx = 0
        self.print_iter = print_iter
        self.tot_nbr_steps = tot_nbr_steps

    def has_stat(self, name):
        return name in self.stats

    def register(self, name, stat_info, ma_weight=0.001):
        if self.has_stat(name):
            sys.exit("Stat already exists")

        if stat_info['type'] == 'avg':
            stat_obj = AvgStat(name, stat_info['freq'], ma_weight=ma_weight)
        else:
            sys.exit("Stat type not supported")

        stat = {'obj': stat_obj, 'name': name, 'type': stat_info['type']}
        self.stats[name] = stat

    def s(self, name):
        return self.stats[name]['obj']

    def next_step(self):
        self.step_idx += 1

    def next_ep(self):
        self.ep_idx += 1
        for stat_name, stat in self.stats.items():
            stat['obj'].next_ep()
        if self.ep_idx % self.print_iter == 0:
            self.print()
            self._plot_to_hdock()

    def print(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].print()

    def plot(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].plot(save_dir=self.log_dir)

    def save(self):
        for stat_name, stat in self.stats.items():
            stat['obj'].save(save_dir=self.log_dir)

def get_random_crop(H, W, crop_size, coords=None, min_frame=0, H_MIN=0, W_MIN=0):
    coords_inside = np.zeros((0, 2))
    for attempt in range(250):
        start_h = random.randint(H_MIN, H - crop_size - 1)
        end_h = start_h + crop_size
        start_w = random.randint(W_MIN, W - crop_size - 1)
        end_w = start_w + crop_size
        if coords is None:
            return [start_h, end_h, start_w, end_w], None, None
        else:
            crop_found = False
            idxs_inside = []
            for i in range(coords.shape[0]):
                coord_h = coords[i, 0]
                coord_w = coords[i, 1]
                if start_h + min_frame <= coord_h and end_h - min_frame > coord_h+1 and start_w + min_frame <= coord_w and end_w - min_frame > coord_w+1:
                    crop_found = True
                    coords_inside = np.concatenate((coords_inside, coords[i, :][np.newaxis, :]), axis=0)
                    idxs_inside.append(i)
            if crop_found:
                return [start_h, end_h, start_w, end_w], coords_inside, np.array(idxs_inside)
    return None, None, None

def _mlp_post_filter(pred_map_binary_list, pred_map_binary_thin_list, pred_map, thresh_thin_cloud, post_filt_sz):
	if post_filt_sz == 1:
		return pred_map_binary_list, pred_map_binary_thin_list
	H, W = pred_map.shape
	for list_idx, pred_map_binary in enumerate(pred_map_binary_list):
		tmp_map = np.zeros_like(pred_map)
		tmp_map_thin = np.zeros_like(pred_map)
		count_map = np.zeros_like(pred_map)
		for i_start in range(post_filt_sz):
			for j_start in range(post_filt_sz):
				for i in range(i_start, H // post_filt_sz):
					for j in range(j_start, W // post_filt_sz):
						count_map[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
						curr_patch = pred_map_binary[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz]
						curr_patch_thin = pred_map_binary_thin_list[min(list_idx, len(thresh_thin_cloud) - 1)][i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz]
						if np.count_nonzero(curr_patch) >= np.prod(curr_patch.shape) // 2:
							tmp_map[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
						if np.count_nonzero(curr_patch_thin) >= np.prod(curr_patch_thin.shape) // 2:
							tmp_map_thin[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
		tmp_map[count_map == 0] = 0
		count_map[count_map == 0] = 1
		tmp_map /= count_map
		assert np.min(tmp_map) >= 0 and np.max(tmp_map) <= 1
		pred_map_binary = tmp_map >= 0.50
		pred_map_binary_list[list_idx] = pred_map_binary

		tmp_map_thin[count_map == 0] = 0
		tmp_map_thin /= count_map
		assert np.min(tmp_map_thin) >= 0 and np.max(tmp_map_thin) <= 1
		pred_map_binary_thin = tmp_map_thin >= 0.50
		pred_map_binary_thin_list[min(list_idx, len(thresh_thin_cloud) - 1)] = pred_map_binary_thin

		# 'Aliasing effect' after this filtering can cause BOTH thin and regular cloud to be active at the same time -- give prevalence to regular
		pred_map_binary_thin_list[0][pred_map_binary_list[0]] = 0

	return pred_map_binary_list, pred_map_binary_thin_list

# Setup MLP-computation function
def mlp_inference(img, means, stds, models, batch_size, thresh_cloud, thresh_thin_cloud, post_filt_sz, device='cpu', predict_also_cloud_binary=False):
	H, W, input_dim = img.shape
	img_torch = torch.reshape((torch.Tensor(img).to(device) - means) / stds, [H * W, input_dim])
	pred_map_tot = 0.0
	pred_map_binary_tot = 0.0
	for model in models:
		pred_map = np.zeros(H * W)
		pred_map_binary = np.zeros(H * W)
		for i in range(0, H * W, batch_size):
			curr_pred = model(img_torch[i : i + batch_size, :])
			pred_map[i : i + batch_size] = curr_pred[:, 0].cpu().detach().numpy()
			if predict_also_cloud_binary:
				pred_map_binary[i : i + batch_size] = curr_pred[:, 1].cpu().detach().numpy()
		pred_map = np.reshape(pred_map, [H, W])
		if predict_also_cloud_binary:
			pred_map_binary = np.reshape(expit(pred_map_binary), [H, W]) >= 0.5
		else:
			pred_map_binary = np.zeros_like(pred_map)#pred_map >= thresh_cloud[-1] <<--- overwritten anyway
			
		# Average model predictions
		pred_map_tot += pred_map / len(models)
		pred_map_binary_tot += pred_map_binary.astype(float) / len(models)
		
	# Return final predictions
	pred_map = pred_map_tot
	if predict_also_cloud_binary:
		pred_map_binary = pred_map_binary_tot >= 0.5
	else:
		pred_map_binary_list = []
		pred_map_binary_thin_list = []
		for thresh in thresh_cloud:
			pred_map_binary_list.append(pred_map_tot >= thresh)
		for thresh in thresh_thin_cloud:
			# Below: A thin cloud is a thin cloud only if it is above the thin thresh AND below the regular cloud thresh
			pred_map_binary_thin_list.append(np.logical_and(pred_map_tot >= thresh, pred_map_tot < thresh_cloud[0]))

	# Potentially do post-processing on the cloud/not cloud (binary)
	# prediction, so that it becomes more spatially coherent
	pred_map_binary_list, pred_map_binary_thin_list = _mlp_post_filter(pred_map_binary_list, pred_map_binary_thin_list, pred_map, thresh_thin_cloud, post_filt_sz)

	# Return
	return pred_map, pred_map_binary_list, pred_map_binary_thin_list

# Compute mIoU
# See https://datascience.stackexchange.com/questions/104746/is-there-an-official-procedure-to-compute-miou
# which says there is no 'official' way to compute mIoU. Either one can do it by computing the mIoU per image
# in a dataset, and then at the end just averaging the per-image-mIoUs (i.e. divide the sum of the mIoUs with
# the number of images in the dataset). OR one can rather keep track of a 'global' estimate across the whole
# dataset, where one then also needs to count the occurrence of each class per image, so that a proper
# weighted average across the whole dataset may be taken in the end. As this is more complicated, we opt here
# for the per-image-uniform-weighted approach. Finally, note that the difference between the two
# approaches will diminish with increasing batch size.
def mIoU(map_pred_flat, gt_flat, sc, nbr_classes, mode=''):
	if mode != '':
		mode = '_' + mode

	# Keep track of the class-specific IoUs and compute the mIoU
	unqs = np.unique(gt_flat)
	assert np.all(unqs >= 0)
	for unq in unqs:
		iou = jaccard_score(gt_flat == unq, map_pred_flat == unq, average='binary')
		sc.s(('IoU_%d' % unq) + mode).collect(iou)

	# NOTE: For the mIoU_true stat, it's the 'last' value printed in the statcollector
	# output that is the correct value! This is because the 'last' value here
	# represents the average of the num_classes individual ious, WHERE THOSE ious
	# are already 'globally-averaged'. Finally, note that we ONLY collect
	# this stat when we have at least one IoU tracked for each of the
	# nbr_classes classes in the dataset
	means = []
	mas = []
	for cls_idx in range(nbr_classes):
		curr_means = sc.s(('IoU_%d' % cls_idx) + mode).get_data()['means']
		curr_mas = sc.s(('IoU_%d' % cls_idx) + mode).get_data()['mas']
		if len(curr_means) == 0:
			return sc
		means.append(curr_means[-1])
		mas.append(curr_mas[-1])
	sc.s(f'mIoU-glob-mean{mode}').collect(np.mean(means))
	sc.s(f'mIoU-exp-ma{mode}').collect(np.mean(mas))
	return sc

def eval_swe_forest_cls(base_path, split='train'):
	all_binary_preds = np.load('skogs_preds.npy')

	if split == 'train':
		all_binary_gts = np.load(os.path.join(base_path, 'skogs_gts_train.npy'))
	elif split == 'val':
		all_binary_gts = np.load(os.path.join(base_path, 'skogs_gts_val.npy'))
	elif split == 'trainval':
		all_binary_gts = np.load(os.path.join(base_path, 'skogs_gts_train.npy'))
		all_binary_gts = np.concatenate([all_binary_gts, np.load(os.path.join(base_path, 'skogs_gts_val.npy'))])
	elif split == 'test':
		all_binary_gts = np.load(os.path.join(base_path, 'skogs_gts_test.npy'))
	assert len(all_binary_preds) == len(all_binary_gts)

	# Print results
	print(all_binary_preds)
	print(all_binary_gts)
	print("Nbr cloudy gt, nbr images total: %d, %d" % (np.count_nonzero(all_binary_gts == 1), len(all_binary_gts)))
	print("Frac cloudy gt: %.4f" % (np.count_nonzero(all_binary_gts == 1) / len(all_binary_gts)))
	print("Accuracy: %.4f" % (np.count_nonzero(all_binary_preds == all_binary_gts) / len(all_binary_preds)))

	rec_0 = np.count_nonzero(all_binary_preds[all_binary_gts == 0] == all_binary_gts[all_binary_gts == 0]) / np.count_nonzero(all_binary_gts == 0)
	rec_1 = np.count_nonzero(all_binary_preds[all_binary_gts == 1] == all_binary_gts[all_binary_gts == 1]) / np.count_nonzero(all_binary_gts == 1)
	print("Recall (balanced): %.4f" % (0.5 * (rec_0 + rec_1)))
	print("Recall (gt is clear (0)): %.4f" % (rec_0))
	print("Recall (gt is cloudy (1)): %.4f" % (rec_1))

	prec_0 = np.count_nonzero(all_binary_gts[all_binary_preds == 0] == 0) / np.count_nonzero(all_binary_preds == 0)
	prec_1 = np.count_nonzero(all_binary_gts[all_binary_preds == 1] == 1) / np.count_nonzero(all_binary_preds == 1)
	print("Precision (balanced): %.4f" % (0.5 * (prec_0 + prec_1)))
	print("Precision (gt is clear (0)): %.4f" % (prec_0))
	print("Precision (gt is cloudy (1)): %.4f" % (prec_1))

	f1_0 = 2 / (1 / rec_0 + 1 / prec_0)
	f1_1 = 2 / (1 / rec_1 + 1 / prec_1)
	print("F1 score (balanced): %.4f" % (0.5 * (f1_0 + f1_1)))
	print("F1 score (gt is clear (0)): %.4f" % (f1_0))
	print("F1 score (gt is cloudy (1)): %.4f" % (f1_1))

# Simple 5-layer MLP model
class MLP5(nn.Module):
	def __init__(self, input_dim, output_dim=1, hidden_dim=64, apply_relu=True):
		super(MLP5, self).__init__()
		self.lin1 = nn.Linear(input_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim)
		self.lin3 = nn.Linear(hidden_dim, hidden_dim)
		self.lin4 = nn.Linear(hidden_dim, hidden_dim)
		self.lin5 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
		self.apply_relu = apply_relu

	def forward(self, x):
		x1 = self.lin1(x)
		x1 = self.relu(x1)
		x2 = self.lin2(x1)
		x2 = self.relu(x2)
		x3 = self.lin3(x2)
		x3 = self.relu(x3)
		x4 = self.lin4(x3)
		x4 = self.relu(x4)
		x5 = self.lin5(x4)
		if self.apply_relu:
			x5[:, 0] = self.relu(x5[:, 0])  # NB: cloud optical thicknesses cannot be negative
		return x5
