#from __future__ import print_function
import os
import time
import random
import datetime
from shutil import copyfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from utils import StatCollector, mlp_inference, mIoU, MLP5, get_random_crop
import netCDF4

# This script can be used BOTH for evaluating SMHI cloud thickness
# prediction models (but for 0-thin-reg cloud prediction; see below)
# AND for refining models. EVAL_ONLY = True (see flags below) implies
# that no refinement occurs.
#
# OBS: THE SCRIPT DEFAULTS TO EVAL_ONLY MODE!

# Global vars
BASE_PATH_DATA = '../data/kappaset'
MONTHS = ['April', 'May', 'June']
#MONTHS = ['July', 'August', 'September']
SPEC_PATHS = []
for month in MONTHS:
	SPEC_PATHS += [os.path.join(BASE_PATH_DATA, month, img_name) for img_name in sorted(os.listdir(os.path.join(BASE_PATH_DATA, month)))]
BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A' , 'B09', 'B10', 'B11', 'B12']
BASE_PATH_LOG = '../log'
USE_GPU = True  # The code uses small-scale models, GPU doesn't seem to accelerate things actually
SEED = 0
CROP_SZ = None  # Set to integer value instead of None --> images will be cropped (similar to what's done with FCN models in kappa_cloud_train.py)
BATCH_SIZE = 256
LR = 0.0003  # Learning rate
WEIGHT_DECAY = 0  # Parameter for ADAM optimizer
BETA1 = 0.9  # Parameter for ADAM optimizer
DO_PLOT = True
PLOT_OUTLET = 'internal'  # 'internal', 'article', or 'poster'
SKIP_BAND_10 = False  # True --> Skip band 10 as input (may be needed for Skogsstyrelsen data)
SKIP_BAND_1 = True  # True --> Skip band 1 as input (done by default due to un-trustworthy in smhi data)
PROPERTY_COLUMN_MAPPING = {'spec_bands': [i for i in range(1, 14)], 'angles': [14, 15, 16], 'thick': [17], 'type': [18], 'prof_id': [19], 'gas_vapour': [20, 21], 'surf_prof': [22]}
INPUTS = ['spec_bands']#, 'angles', 'gas_vapour', 'surf_prof']  # Add keys from PROPERTY_COLUMN_MAPPING to use those as inputs
THRESHOLD_THICKNESS_IS_CLOUD = 0.025
THRESHOLD_THICKNESS_IS_THIN_CLOUD = 0.015
CLASSES_AS_NOT_PART_OF_CLEAR = [0, 5]  # [] --> ALL of the 'MISSING (0)', 'CLOUD SHADOW (2)', 'UNDEFINED (5)' and 'CLEAR (1)' are lumped together as a single 'clear' category
# OBS: according to https://zenodo.org/record/5095024, it seems as that 0 and 5 should be 'flipped' i.e. it seems 5 is the missing one, and 0 is the unknown one
MLP_POST_FILTER_SZ = 2  # 1 --> no filtering, >= 2 --> majority vote within that-sized square
USE_SLACK_NO_CLOUD_LOSS = False  # True --> no-cloud loss is zero if predicted below thin-threshold (results better with False here)
EVAL_ONLY = False  # True --> No backprop of model, only evaluated
MODEL_LOAD_PATH = None

assert not (isinstance(MODEL_LOAD_PATH, list) and not EVAL_ONLY)

if not isinstance(MODEL_LOAD_PATH, list):
	MODEL_LOAD_PATH = [MODEL_LOAD_PATH]
if not isinstance(THRESHOLD_THICKNESS_IS_CLOUD, list):
	THRESHOLD_THICKNESS_IS_CLOUD = [THRESHOLD_THICKNESS_IS_CLOUD]
if not isinstance(THRESHOLD_THICKNESS_IS_THIN_CLOUD, list):
	THRESHOLD_THICKNESS_IS_THIN_CLOUD = [THRESHOLD_THICKNESS_IS_THIN_CLOUD]

# Specify model input and output dimensions
input_dim = np.sum([len(PROPERTY_COLUMN_MAPPING[inp]) for inp in INPUTS]) - SKIP_BAND_10 - SKIP_BAND_1
output_dim = 1  # + PREDICT_ALSO_CLOUD_BINARY

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)
copyfile("kappa_cloud_opt_thick.py", os.path.join(log_dir, "kappa_cloud_opt_thick.py"))

# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(SEED)

# Set max number of threads to use to 10 (dgx1 limit)
torch.set_num_threads(10)

# Setup prediction model
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
models = []
for model_load_path in MODEL_LOAD_PATH:
	model = MLP5(input_dim, output_dim, apply_relu=EVAL_ONLY)
	if model_load_path is not None:
		model.load_state_dict(torch.load(model_load_path, map_location=device))
	model.to(device)
	model.eval()
	models.append(model)
	
# Setup (refinement) loss and optimizer
criterion_MSE = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.999))

# Setup statistics collector
sc = StatCollector(stat_train_dir, 999999, 10)
sc.register('IoU_0', {'type': 'avg', 'freq': 'step'})
sc.register('IoU_1', {'type': 'avg', 'freq': 'step'})
sc.register('IoU_2', {'type': 'avg', 'freq': 'step'})
sc.register('mIoU-glob-mean', {'type': 'avg', 'freq': 'step'})
sc.register('mIoU-exp-ma', {'type': 'avg', 'freq': 'step'})

sc.register('acc', {'type': 'avg', 'freq': 'step'})

sc.register('recall-cloud-reg', {'type': 'avg', 'freq': 'step'})
sc.register('recall-cloud-thin', {'type': 'avg', 'freq': 'step'})
sc.register('recall-no-cloud', {'type': 'avg', 'freq': 'step'})

sc.register('precision-cloud-reg', {'type': 'avg', 'freq': 'step'})
sc.register('precision-cloud-thin', {'type': 'avg', 'freq': 'step'})
sc.register('precision-no-cloud', {'type': 'avg', 'freq': 'step'})

sc.register('loss', {'type': 'avg', 'freq': 'step'})
sc.register('loss-nocloud', {'type': 'avg', 'freq': 'step'})
sc.register('loss-regular', {'type': 'avg', 'freq': 'step'})
sc.register('loss-thin', {'type': 'avg', 'freq': 'step'})

# Specific means-stds of dataset (based on SMHI synthetic)
means = torch.Tensor(np.array([0.64984976, 0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])).to(device)
stds = torch.Tensor(np.array([0.3596485, 0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])).to(device)

if SKIP_BAND_10:
	means = means[[0,1,2,3,4,5,6,7,8,9,11,12]]
	stds = stds[[0,1,2,3,4,5,6,7,8,9,11,12]]
	BANDS.remove('B10')
if SKIP_BAND_1:
	means = means[1:]
	stds = stds[1:]
	BANDS.remove('B01')

# Perform initial data shuffling
perm = list(np.random.permutation(len(SPEC_PATHS)))
SPEC_PATHS = [SPEC_PATHS[p] for p in perm]

print("Starting loop...")
if EVAL_ONLY:
	model.eval()
else:
	model.train()
tot_ctr = 0
img_idx = 0
if EVAL_ONLY:
	TOT_NBR_ITERS = len(SPEC_PATHS)
else:
	TOT_NBR_ITERS = 100000
while tot_ctr <= TOT_NBR_ITERS:
	if img_idx >= len(SPEC_PATHS):
		img_idx = 0
	spec_path = SPEC_PATHS[img_idx]
	
	# Read image and gt
	file2read = netCDF4.Dataset(spec_path,'r')
	layers = []
	for i, band in enumerate(BANDS):
		layers.append(file2read.variables[band][:][:, :, np.newaxis])
	img = np.concatenate(layers, axis=2)
	gt = file2read.variables['Label'][:]
	assert np.min(gt) >= 0 and np.max(gt)<= 5
	H, W, C = img.shape
	img *= 7

	if CROP_SZ is not None:
		gt_cloud_thin = (gt == 3) + 0.0
		gt_cloud_regular = (gt == 4) + 0.0
		gt_cloud = 2 * gt_cloud_regular + gt_cloud_thin
		# The below for-loop sets -1 on those pixels which
		# we do NOT want to be part of the 'no cloud' class
		for cls_idx in CLASSES_AS_NOT_PART_OF_CLEAR:
			gt_cloud[gt == cls_idx] = -1
		gt_cloud = np.array(gt_cloud)  # regular array from masked array
		
		# Get a random crop
		CROP_SZ = 128
		crop_coords = get_random_crop(H, W, CROP_SZ)[0]
		curr_gt_cloud = gt_cloud[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]

		# Ensure crop always thin cloud (if possible), or otherwise thick cloud (if possible)
		MAX_NBR_TRYS = 51
		if np.any(gt_cloud == 1):  # include thin cloud
			try_ctr = 0
			while not np.any(curr_gt_cloud == 1) and try_ctr < MAX_NBR_TRYS:
				crop_coords = get_random_crop(H, W, CROP_SZ)[0]
				curr_gt_cloud = gt_cloud[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
				try_ctr += 1
		elif np.any(gt_cloud == 2):  # include thick cloud
			try_ctr = 0
			while not np.any(curr_gt_cloud == 2) and try_ctr < MAX_NBR_TRYS:
				crop_coords = get_random_crop(H, W, CROP_SZ)[0]
				curr_gt_cloud = gt_cloud[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
				try_ctr += 1

		# Perform cropping of all relevant entities
		H, W, C = img.shape
		img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
		gt_cloud = curr_gt_cloud
		gt_cloud_thin = gt_cloud_thin[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
		gt_cloud_regular = gt_cloud_regular[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]

	# Perform training-based prediction on subset of image, then compute loss
	if not EVAL_ONLY:
		# This part creates a balanced batch of thin cloud | regular cloud | no cloud
		gt_binary_reshaped = np.reshape(gt, [H*W])
		gt_binary_nnz_idxs = np.nonzero(gt_binary_reshaped == 3)[0]  # thin cloud
		gt_binary_nnz2_idxs = np.nonzero(gt_binary_reshaped == 4)[0]  # regular cloud
		gt_not_cloud = np.logical_and(gt_binary_reshaped != 3, gt_binary_reshaped != 4)
		# The below for-loop ensures that some (potentially) unwanted categories
		# are removed from the cloud-free class (e.g. missing or unknown pixels)
		for cls_idx in CLASSES_AS_NOT_PART_OF_CLEAR:
			gt_not_cloud = np.logical_and(gt_not_cloud, gt_binary_reshaped != cls_idx)
		gt_binary_z_idxs = np.nonzero(gt_not_cloud)[0]  # no cloud
		if len(gt_binary_nnz_idxs) < BATCH_SIZE//4 or len(gt_binary_nnz2_idxs) < BATCH_SIZE//4 or len(gt_binary_z_idxs) < BATCH_SIZE//2:
			img_idx += 1
			continue
		idxs_chosen_nnz = np.random.choice(gt_binary_nnz_idxs, size=BATCH_SIZE//4, replace=False)
		idxs_chosen_nnz2 = np.random.choice(gt_binary_nnz2_idxs, size=BATCH_SIZE//4, replace=False)
		idxs_chosen_z = np.random.choice(gt_binary_z_idxs, size=BATCH_SIZE//2, replace=False)
		idxs_chosen = np.concatenate([idxs_chosen_nnz, idxs_chosen_nnz2, idxs_chosen_z])
		
		# Perform model prediction
		img_reshaped = np.reshape(img, [H * W, input_dim])
		img_torch = (torch.Tensor(img_reshaped).to(device) - means) / stds
		gts_torch = torch.Tensor(gt_binary_reshaped).to(device)
		curr_gts_binary = (gts_torch[idxs_chosen] == 3) + (gts_torch[idxs_chosen] == 4) + 0.0
		curr_gts_nocloud = 1 - curr_gts_binary
		curr_gts_regular = (gts_torch[idxs_chosen] == 4) + 0.0
		curr_gts_thin = (gts_torch[idxs_chosen] == 3) + 0.0
		preds = model(img_torch[idxs_chosen, :])

		# Compute loss
		if USE_SLACK_NO_CLOUD_LOSS:
			loss_no_cloud = criterion_MSE(nn.ReLU()(preds[curr_gts_nocloud == 1] - THRESHOLD_THICKNESS_IS_THIN_CLOUD[0]),
										  torch.zeros_like(preds[curr_gts_nocloud == 1]).to(device))
		else:
			# Empirically results get better by stronger enforcement of the no-cloud to aim for exactly 0 thickness
			loss_no_cloud = criterion_MSE(preds[curr_gts_nocloud == 1],
										  torch.zeros_like(preds[curr_gts_nocloud == 1]).to(device))
		loss_regular = criterion_MSE(nn.ReLU()(THRESHOLD_THICKNESS_IS_CLOUD[0] - preds[curr_gts_regular == 1]),
									 torch.zeros_like(preds[curr_gts_regular == 1]).to(device))				  
		loss_thin = criterion_MSE(nn.ReLU()(THRESHOLD_THICKNESS_IS_THIN_CLOUD[0] - preds[curr_gts_thin == 1]) +
								  nn.ReLU()(preds[curr_gts_thin == 1] - THRESHOLD_THICKNESS_IS_CLOUD[0]),
								  torch.zeros_like(preds[curr_gts_thin == 1]).to(device))
		factor1 = 1/3
		factor2 = 1/3
		factor3 = 1/3
		loss = factor1 * loss_no_cloud + factor2 * loss_regular + factor3 * loss_thin
		
		# Track stats
		sc.s('loss').collect(loss.cpu().detach().numpy())
		sc.s('loss-nocloud').collect(loss_no_cloud.cpu().detach().numpy())
		sc.s('loss-regular').collect(loss_regular.cpu().detach().numpy())
		sc.s('loss-thin').collect(loss_thin.cpu().detach().numpy())
		
		# Update model weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# Perform model prediction (full img) -- for eval purposes
	do_full_pred = EVAL_ONLY or img_idx % 10 == 0
	if do_full_pred:
		pred_map, pred_map_binary_list, pred_map_binary_thin_list = mlp_inference(img, means, stds, models, BATCH_SIZE,
																				  THRESHOLD_THICKNESS_IS_CLOUD,
																				  THRESHOLD_THICKNESS_IS_THIN_CLOUD,
																				  MLP_POST_FILTER_SZ, device)
		
		# Compute various accuracies and track stats
		if CROP_SZ is None:
			gt_cloud_thin = (gt == 3) + 0.0
			gt_cloud_regular = (gt == 4) + 0.0
			gt_cloud = 2 * gt_cloud_regular + gt_cloud_thin
			# The below for-loop sets -1 on those pixels which
			# we do NOT want to be part of the 'no cloud' class
			for cls_idx in CLASSES_AS_NOT_PART_OF_CLEAR:
				gt_cloud[gt == cls_idx] = -1

		# Compute mIoU
		# See https://datascience.stackexchange.com/questions/104746/is-there-an-official-procedure-to-compute-miou
		# where it say there is no 'official' way to compute mIoU. Either one can do it by computing the mIoU per image
		# in a dataset, and then at the end just averagin the per-image-mIoUs (i.e. divide the sum of the mIoUs with 
		# the number of images in the dataset). OR one can rather keep track of a 'global' estimate across the whole
		# dataset, where one then also needs to count the occurrence of each class per image, so that a proper
		# weighted average across the whole dataset may be taken in the end. As this is more compliated, we opt here
		# for the per-image-unifrom-weighted approach. Finally, note that the difference between the two
		# approaches will diminish with increasing batch size.
		map_pred_flat = (2*pred_map_binary_list[0] + 0.0 + pred_map_binary_thin_list[0]).flatten()  # make it 1D
		gt_flat = gt_cloud.flatten()  # make it 1D
		
		# Take the -1 category out of the picture
		map_pred_flat = map_pred_flat[gt_flat >= 0]
		gt_flat = gt_flat[gt_flat >= 0]

		# Track IoU and mIoU stats
		sc = mIoU(map_pred_flat, gt_flat, sc, nbr_classes=3)

		acc = np.count_nonzero(map_pred_flat == gt_flat) / H / W
		sc.s('acc').collect(acc)
		if np.any(gt_cloud == 0):
			rec_no_cloud = np.count_nonzero(np.logical_and(pred_map_binary_thin_list[0][gt_cloud == 0] == 0, pred_map_binary_list[0][gt_cloud == 0] == 0)) / np.count_nonzero(gt_cloud == 0)
			sc.s('recall-no-cloud').collect(rec_no_cloud)
		else:
			recall_no_cloud = -1
		if np.any(pred_map_binary_thin_list[0] == 0):
			prec = np.count_nonzero(gt_cloud[pred_map_binary_thin_list[0] == 0] == 0) / np.count_nonzero(pred_map_binary_thin_list[0] == 0)
			sc.s('precision-no-cloud').collect(prec)
		# Track also individual cloud type stats
		if np.any(gt_cloud_regular == 1):
			rec_cloud_reg = np.count_nonzero(pred_map_binary_list[0][gt_cloud_regular == 1] == 1) / np.count_nonzero(gt_cloud_regular)
			sc.s('recall-cloud-reg').collect(rec_cloud_reg)
		else:
			rec_cloud_reg = -1
		if np.any(pred_map_binary_list[0] == 1):
			prec = np.count_nonzero(gt_cloud_regular[pred_map_binary_list[0] == 1] == 1) / np.count_nonzero(pred_map_binary_list[0] == 1)
			sc.s('precision-cloud-reg').collect(prec)
		if np.any(gt_cloud_thin == 1):
			rec_cloud_thin = np.count_nonzero(pred_map_binary_thin_list[0][gt_cloud_thin == 1] == 1) / np.count_nonzero(gt_cloud_thin)
			sc.s('recall-cloud-thin').collect(rec_cloud_thin)
		else:
			rec_cloud_thin = -1
		if np.any(pred_map_binary_thin_list[0] == 1):
			prec = np.count_nonzero(gt_cloud_thin[pred_map_binary_thin_list[0] == 1] == 1) / np.count_nonzero(pred_map_binary_thin_list[0] == 1)
			sc.s('precision-cloud-thin').collect(prec)
				
		# Occassionally print out how things are progressing
		sc.print()
		sc.save()
		print(img_idx, len(SPEC_PATHS), tot_ctr, TOT_NBR_ITERS)

		# Plot results
		if DO_PLOT:
			# Show RGB part of image with predictions + GT
			if SKIP_BAND_1:
				rgb_img = img[:, :, [2,1,0]] / np.max(img[:, :, [2,1,0]])
			else:
				rgb_img = img[:, :, [3,2,1]] / np.max(img[:, :, [3,2,1]])
				
			fig = plt.figure(figsize=(16, 16))
			
			if PLOT_OUTLET == 'poster':

				plt.subplots_adjust(wspace=0.03, hspace=0)
				n_cols = 4
				
				fig.add_subplot(1,n_cols,1)
				plt.imshow(rgb_img)
				#plt.title('image')
				plt.axis('off')

				fig.add_subplot(1,n_cols,2)
				plt.imshow(pred_map)
				#plt.title('pred-thick (rel)')
				plt.axis('off')

				fig.add_subplot(1,n_cols,3)
				# Note: vmax set to 3 to make it 'corresponding' with the ground truth color scale which includes undefined
				plt.imshow(2*pred_map_binary_list[0] + 0.0 + pred_map_binary_thin_list[0], vmin=0, vmax=3)
				#plt.title('pred-thresh, acc: %.3f, rec-no-cloud: %.3f,\n rec-reg: %.3f, rec-thin: %.3f' % (acc, rec_no_cloud, rec_cloud_reg, rec_cloud_thin))
				plt.axis('off')

				fig.add_subplot(1,n_cols,4)
				gt_cloud[gt_cloud == -1] = 3  # fully yellow = 'unconsidered' class
				plt.imshow(gt_cloud, vmin=0, vmax=3)
				#plt.title('gt-cloud')
				plt.axis('off')

				#fig.add_subplot(2,3,4)
				#plt.imshow(pred_map)
				#plt.title('pred-thick (rel)')
				#plt.axis('off')

				#fig.add_subplot(2,3,5)
				#plt.imshow(pred_map, vmin=0, vmax=1)
				#plt.title('pred-thick (abs)')
				#plt.axis('off')
			
			elif PLOT_OUTLET == 'article':
				
				plt.subplots_adjust(wspace=0.03, hspace=0)
				n_cols = 4
				
				fig.add_subplot(1,n_cols,1)
				plt.imshow(rgb_img)
				plt.title('image')
				plt.axis('off')

				fig.add_subplot(1,n_cols,2)
				# Note: vmax set to 3 to make it 'corresponding' with the ground truth color scale which includes undefined
				plt.imshow(2*pred_map_binary_list[0] + 0.0 + pred_map_binary_thin_list[0], vmin=0, vmax=3)
				plt.title('pred-thresh, acc: %.3f, rec-no-cloud: %.3f,\n rec-reg: %.3f, rec-thin: %.3f' % (acc, rec_no_cloud, rec_cloud_reg, rec_cloud_thin))
				plt.axis('off')

				fig.add_subplot(1,n_cols,4)
				gt_cloud[gt_cloud == -1] = 3  # fully yellow = 'unconsidered' class
				plt.imshow(gt_cloud, vmin=0, vmax=3)
				plt.title('gt-cloud')
				plt.axis('off')

				fig.add_subplot(1,n_cols,3)
				plt.imshow(pred_map)
				plt.title('pred-thick (rel)')
				plt.axis('off')

				#fig.add_subplot(2,3,5)
				#plt.imshow(pred_map, vmin=0, vmax=1)
				#plt.title('pred-thick (abs)')
				#plt.axis('off')

			elif PLOT_OUTLET == 'internal':
				fig.add_subplot(2,3,1)
				plt.imshow(rgb_img)
				plt.title('image')
				plt.axis('off')

				fig.add_subplot(2,3,2)
				plt.imshow(gt, vmin=0, vmax=5)
				plt.title('gt')
				plt.axis('off')
				
				fig.add_subplot(2,3,3)
				gt_cloud[gt_cloud == -1] = 3  # fully yellow = 'unconsidered' class
				plt.imshow(gt_cloud, vmin=0, vmax=3)
				plt.title('gt-cloud')
				plt.axis('off')

				fig.add_subplot(2,3,4)
				plt.imshow(pred_map)
				plt.title('pred-thick (rel)')
				plt.axis('off')

				fig.add_subplot(2,3,5)
				plt.imshow(pred_map, vmin=0, vmax=1)
				plt.title('pred-thick (abs)')
				plt.axis('off')

				fig.add_subplot(2,3,6)
				# Note: vmax set to 3 to make it 'corresponding' with the ground truth color scale which includes undefined
				plt.imshow(2*pred_map_binary_list[0] + 0.0 + pred_map_binary_thin_list[0], vmin=0, vmax=3)
				plt.title('pred-thresh, acc: %.3f, rec-no-cloud: %.3f,\n rec-reg: %.3f, rec-thin: %.3f' % (acc, rec_no_cloud, rec_cloud_reg, rec_cloud_thin))
				plt.axis('off')

			plt.savefig(os.path.join(stat_train_dir, 'preds_%d.png' % img_idx))
			plt.cla()
			plt.clf()
			plt.close('all')

	# Increment counters
	tot_ctr += 1
	img_idx += 1
	
	if not EVAL_ONLY and tot_ctr > 100 and tot_ctr % 15000 == 0:
		# Occassionally save model weights
		torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % tot_ctr))

# After training, save model weights
if not EVAL_ONLY:
	print("Saving model weights")
	torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % TOT_NBR_ITERS))
	print("DONE!")

print("DONE!")
