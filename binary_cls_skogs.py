#from __future__ import print_function
import os
import time
import random
import datetime
from shutil import copyfile
import tifffile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
import xarray as xr
import numpy as np
import json
from utils import StatCollector


BASE_PATH_LOG = '../log'
BASE_PATH_DATA = '../data/skogsstyrelsen/'
BATCH_SIZE = 64  # Batch size during model training
NUM_TRAIN_BATCHES = 1000  # How many batches to train model for
IM_H_BATCH = 64  # Need to upscale images for ResNet18
IM_W_BATCH = 64  # Need to upscale images for ResNet18
LR = 0.0002  # Learning rate
WEIGHT_DECAY = 0  # Parameter for ADAM optimizer
BETA1 = 0.9  # Parameter for ADAM optimizer
SEED = 0
MODEL_LOAD_PATH = None
SPLIT_TO_USE = 'test'  # 'train', 'val', 'trainval' or 'test'
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DO_PLOT = False
EVAL_ONLY = False  # True --> No backprop of model, only evaluated.
SKIP_BAND_10 = True  # True --> Skip band 10 as input (may be needed for Skogsstyrelsen data)
SKIP_BAND_1 = True  # True --> Skip band 1 as input (typicall done if trained on SMHI)
BAND_NAMES = ['b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b8a', 'b09', 'b10', 'b11', 'b12']
SCL_COLORS = {0: np.array([0, 0, 0]), #No Data (black)
             1: np.array([255, 0, 0]), # Saturated or defective pixel (red)
             2: np.array([47, 47, 47]), # Topographic casted shadows (almost black)
             3: np.array([100, 50, 0]), # Cloud shadows (dark brown)
             4: np.array([0, 160, 0]), # Vegetation (green)
             5: np.array([255, 230, 90]), # Not-vegetated (orange-yellow)
             6: np.array([0, 0, 255]), # Water (blue)
             7: np.array([128, 128, 128]), # Unclassified (bluish-purpulish)
             8: np.array([192, 192, 192]), # Cloud medium probability (gray)
             9: np.array([255, 255, 255]), # Cloud high probability (white)
             10: np.array([100, 200, 255]), # Thin cirrus (light bluish)
             11: np.array([255, 150, 255])} # Snow or ice (pink)
assert not (EVAL_ONLY and BATCH_SIZE > 1)
assert IM_W_BATCH == IM_H_BATCH

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)
copyfile("binary_cls_skogs.py", os.path.join(log_dir, "binary_cls_skogs.py"))

def color_scl_correctly(scl_layer):
	# Function for setting the correct colors so that they match
	# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
	scl_layer_3chan = np.zeros((scl_layer.shape[0], scl_layer.shape[1], 3), dtype=int)
	for key, value in SCL_COLORS.items():
		if key > 0:  # <-- not needed for key=0 since scl_layer_3chan initialized as zeros
			scl_layer_3chan[scl_layer == key, :] = value
	return scl_layer_3chan

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set max number of threads to use to 10 (dgx1 limit)
torch.set_num_threads(10)

# Read data + corresponding json info (incl ground truth)
img_paths_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_train.npy')))
img_paths_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_val.npy')))
img_paths_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_test.npy')))
json_content_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_train.npy'), allow_pickle=True))
json_content_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_val.npy'), allow_pickle=True))
json_content_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_test.npy'), allow_pickle=True))

# Means, stds from synthetic data
means = torch.Tensor(np.array([0.64984976, 0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])).to(DEVICE)  # based on synth
stds = torch.Tensor(np.array([0.3596485, 0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])).to(DEVICE)  # based on synth

if SKIP_BAND_10:
	means = means[[0,1,2,3,4,5,6,7,8,9,11,12]]
	stds = stds[[0,1,2,3,4,5,6,7,8,9,11,12]]
	BAND_NAMES.remove('b10')
if SKIP_BAND_1:
	means = means[1:]
	stds = stds[1:]
	BAND_NAMES.remove('b01')

# Setup simple resnet18 for the binary classification task
input_dim = 13 - SKIP_BAND_10 - SKIP_BAND_1
output_dim = 1
model = models.resnet18(num_classes=output_dim)
model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3,bias=False)
if MODEL_LOAD_PATH is not None:
	model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
model.to(DEVICE)

# Setup loss and optimizer
criterion_binary = nn.BCEWithLogitsLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.999))

# Setup statistics collector
sc = StatCollector(stat_train_dir, NUM_TRAIN_BATCHES, 10)
sc.register('BCE_loss', {'type': 'avg', 'freq': 'step'})
sc.register('acc', {'type': 'avg', 'freq': 'step'})
sc.register('rec-non-corr', {'type': 'avg', 'freq': 'step'})
sc.register('rec-corr', {'type': 'avg', 'freq': 'step'})
sc.register('prec-non-corr', {'type': 'avg', 'freq': 'step'})
sc.register('prec-corr', {'type': 'avg', 'freq': 'step'})
sc.register('frac-corr', {'type': 'avg', 'freq': 'step'})

# Run model on desired split
if SPLIT_TO_USE == 'train':
	img_paths = img_paths_train
	json_paths = json_content_train
elif SPLIT_TO_USE == 'val':
	img_paths = img_paths_val
	json_paths = json_content_val
elif SPLIT_TO_USE == 'trainval':
	img_paths = img_paths_train + img_paths_val
	json_paths = json_content_train + json_content_val
elif SPLIT_TO_USE == 'test':
	img_paths = img_paths_test
	json_paths = json_content_test
tot_ctr = 0
img_idx = 0
outer_loop_active = True
if EVAL_ONLY:
	model.eval()
while tot_ctr <= NUM_TRAIN_BATCHES:

	# Setup batch containers
	img_batch = torch.zeros((BATCH_SIZE, input_dim, IM_H_BATCH, IM_W_BATCH)).to(DEVICE)
	gt_batch = torch.zeros([BATCH_SIZE]).to(DEVICE)

	# Create batch
	nbr_corrupts = 0
	nbr_non_corrupts = 0
	batch_idx = 0
	while batch_idx < BATCH_SIZE:

		# Re-shuffle data if reached end of epoch
		if img_idx >= len(img_paths):
			if EVAL_ONLY:
				outer_loop_active = False
				break
			perm = list(np.random.permutation(len(img_paths)))
			img_paths = [img_paths[p] for p in perm]
			json_paths = [json_paths[p] for p in perm]
			img_idx = 0

		# Below balances batches based on GT (1=corrupt, 0=not corrupt)
		gt_corrupt = int(json_paths[img_idx]['MolnDis'])
		if nbr_corrupts < nbr_non_corrupts and not gt_corrupt:
			img_idx += 1
			continue
		elif nbr_corrupts > nbr_non_corrupts and gt_corrupt:
			img_idx += 1
			continue
		nbr_corrupts += gt_corrupt
		nbr_non_corrupts += (1 - gt_corrupt)

		# Extract date to see if data is from before or after Jan 2022
		# (this affects the normalization used for the image)
		img_path = img_paths[img_idx]
		img = xr.open_dataset(img_path)
		yy_mm_dd = getattr(img, 'time').values[0]
		yy = yy_mm_dd.astype('datetime64[Y]').astype(int) + 1970
		mm = yy_mm_dd.astype('datetime64[M]').astype(int) % 12 + 1

		# Setup and normalize image
		band_list = []
		for band_name in BAND_NAMES:
			if yy >= 2022 and mm >= 1:  # New normalization after Jan 2022
				band_list.append((getattr(img, band_name).values - 1000) / 10000)  # -1k and then 10k division
			else:
				band_list.append(getattr(img, band_name).values / 10000)  # 10k division
		img = np.concatenate(band_list, axis=0)
		img = np.transpose(img, [1,2,0])
		scl_layer_skogs = np.squeeze(getattr(xr.open_dataset(img_path), 'scl').values)

		# Need to rotate and/or mirror things correctly
		img = np.fliplr(img).copy()
		img = np.flipud(img).copy()
		scl_layer_skogs = np.fliplr(scl_layer_skogs)
		scl_layer_skogs = np.flipud(scl_layer_skogs)
		scl_layer_raw = scl_layer_skogs
		scl_layer = color_scl_correctly(scl_layer_raw)

		# Data aug during training
		if not EVAL_ONLY:
			if random.choice([False, True]):
				img = np.flip(img, axis=0)
			if random.choice([False, True]):
				img = np.flip(img, axis=1)

		# Normalize, reshape, torchify and insert to batch
		img_torch = torch.transpose(torch.transpose((torch.Tensor(img.copy()[np.newaxis, :, :, :]).to(DEVICE) - means) / stds, 1, 3), 2, 3)
		img_torch = transforms.functional.resize(img_torch, [IM_H_BATCH, IM_W_BATCH])
		img_batch[batch_idx, :, :, :] = img_torch

		# Read GT (1=corrupt, 0=not corrupt), torchify and insert to batch
		molndis = json_paths[img_idx]['MolnDis']
		gt_torch = torch.Tensor([int(molndis)]).to(DEVICE)
		gt_batch[batch_idx] = gt_torch

		# Update local counters
		batch_idx += 1
		img_idx += 1

	if not outer_loop_active:
		break

	# Perform prediction
	if BATCH_SIZE > 1:
		preds = torch.squeeze(model(img_batch))
	else:
		preds = model(img_batch)[0,:]

	# Compute loss and do backprop
	loss = criterion_binary(preds, gt_batch)
	if not EVAL_ONLY:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# Track stats
	sc.s('BCE_loss').collect(loss.cpu().detach().numpy())
	preds_bin = torch.sigmoid(preds) >= 0.5
	corrects = (preds_bin == gt_batch).float()
	sc.s('acc').collect(torch.mean(corrects).cpu().detach().numpy())
	if torch.any(gt_batch == 0):
		sc.s('rec-non-corr').collect(torch.mean(corrects[gt_batch == 0]).cpu().detach().numpy())
	if torch.any(gt_batch == 1):
		sc.s('rec-corr').collect(torch.mean(corrects[gt_batch == 1]).cpu().detach().numpy())
	if torch.any(preds_bin == 0):
		xxx = torch.count_nonzero(gt_batch[preds_bin == 0] == 0) / torch.count_nonzero(preds_bin == 0)
		sc.s('prec-non-corr').collect(xxx.cpu().detach().numpy())
	if torch.any(preds_bin == 1):
		xxx = torch.count_nonzero(gt_batch[preds_bin == 1] == 1) / torch.count_nonzero(preds_bin == 1)
		sc.s('prec-corr').collect(xxx.cpu().detach().numpy())
	sc.s('frac-corr').collect(torch.mean(gt_batch).cpu().detach().numpy())

	# Occassionally show stats
	if tot_ctr % 10 == 0:
		print(tot_ctr, NUM_TRAIN_BATCHES)
		sc.print()

	# Occassionally save model
	if tot_ctr % 100 == 0 and tot_ctr > 0:
		torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % tot_ctr))

	# Update global counter
	tot_ctr += 1

	# Visualize results
	if DO_PLOT:
		min_img_rgb = np.array([2.00000009e-03, 3.00000014e-04, 9.99999975e-05])
		max_img_rgb = np.array([0.912, 0.90399998, 0.9016])
		nbr_rows = 1
		fig = plt.figure(figsize=(16, 16))
		if SKIP_BAND_1:
			rgb_img = (img[:, :, [2,1,0]] - min_img_rgb) / max_img_rgb
		else:
			rgb_img = img[:, :, [3,2,1]] / np.max(img[:, :, [3,2,1]])

		fig.add_subplot(nbr_rows,2,1)
		plt.imshow(rgb_img, vmin=0, vmax=1)
		plt.title('pred: %d | gt: %s (1=corrupt)' % (pred_corrupt, molndis))
		plt.axis('off')
		
		# Show also SCL band
		fig.add_subplot(nbr_rows,2,2)
		plt.imshow(scl_layer, vmin=0, vmax=255)
		plt.title('scl')
		plt.axis('off')

		plt.savefig(os.path.join(stat_train_dir, '../binary_skogs_%d.png' % (img_idx)))
		plt.cla()
		plt.clf()
		plt.close('all')
		print("PLOTTED", img_idx)

# After training, save model weights
if not EVAL_ONLY:
	print("Saving model weights")
	torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % NUM_TRAIN_BATCHES))

print("DONE BINARY SKOGS")
