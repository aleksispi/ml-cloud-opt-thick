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
from fcnpytorch.torchfcn.models import FCN8s as FCN8s
from utils import StatCollector, get_random_crop, mIoU
import netCDF4

# Global vars
BASE_PATH_DATA = '../data/kappaset'
MONTHS = ['April', 'May', 'June']
#MONTHS = ['July', 'August', 'September']
SPEC_PATHS = []
for month in MONTHS:
	SPEC_PATHS += [os.path.join(BASE_PATH_DATA, month, img_name) for img_name in sorted(os.listdir(os.path.join(BASE_PATH_DATA, month)))]
BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A' , 'B09', 'B10', 'B11', 'B12']
BASE_PATH_LOG = '../log'
USE_GPU = True
SEED = 0
BATCH_SIZE = 64  # Batch size during model training
LR = 0.0002  # Learning rate
WEIGHT_DECAY = 0  # Parameter for ADAM optimizer
BETA1 = 0.9  # Parameter for ADAM optimizer
IM_H = 128  # Height of image crop
IM_W = 128  # Width of image crop
CLASSES_AS_NOT_PART_OF_CLEAR = [0, 5]  # [] --> ALL of the 'MISSING (0)', 'CLOUD SHADOW (2)', 'UNDEFINED (5)' and 'CLEAR (1)' are lumped together as a single 'clear' category
# OBS: according to https://zenodo.org/record/5095024, it seems as that 0 and 5 should be 'flipped' i.e. it seems 5 is the missing one, and 0 is the unknown one
SKIP_BAND_10 = False  # True --> Skip band 10 as input (may be needed for Skogsstyrelsen data)
SKIP_BAND_1 = True  # True --> Skip band 1 as input (done by default due to un-trustworthy in smhi data)
PROPERTY_COLUMN_MAPPING = {'spec_bands': [i for i in range(1, 14)], 'angles': [14, 15, 16], 'thick': [17], 'type': [18], 'prof_id': [19], 'gas_vapour': [20, 21], 'surf_prof': [22]}
INPUTS = ['spec_bands']#, 'angles', 'gas_vapour', 'surf_prof']  # Add keys from PROPERTY_COLUMN_MAPPING to use those as inputs
MODEL_LOAD_PATH = None  # See examples on how to point to a specific model path below. None --> Model randomly initialized.
EVAL_ONLY = False  # True --> No backprop of model, only evaluated.
NUM_TRAIN_ITER = 1000000  # For how many batches to train the model
DO_PLOT = False
PLOT_FOR_ARTICLE = False  # True --> Special mode with style that is applicable for paper

assert not (EVAL_ONLY and BATCH_SIZE > 1)

# Specify in- and output dimensions
input_dim = np.sum([len(PROPERTY_COLUMN_MAPPING[inp]) for inp in INPUTS]) - SKIP_BAND_10 - SKIP_BAND_1
output_dim = 3

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)
copyfile("kappa_cloud_train.py", os.path.join(log_dir, "kappa_cloud_train.py"))

# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(SEED)

# Set max number of threads to use to 10 (dgx1 limit)
torch.set_num_threads(10)

# Setup prediction model, loss and optimizer
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
model = FCN8s(n_class=output_dim, dim_input=input_dim, weight_init='normal')
if MODEL_LOAD_PATH is not None:
	model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
	
model.to(device)
criterion_CE = nn.CrossEntropyLoss(ignore_index=-1).to(device)  # weight=torch.Tensor(np.array([0.05, 0.475, 0.475])).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.999))

# Setup statistics collector
sc = StatCollector(stat_train_dir, NUM_TRAIN_ITER, 10)
sc.register('CE_loss', {'type': 'avg', 'freq': 'step'})
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
sc.register('frac-img-thin', {'type': 'avg', 'freq': 'step'})
sc.register('frac-img-thick', {'type': 'avg', 'freq': 'step'})
sc.register('frac-thin', {'type': 'avg', 'freq': 'step'})
sc.register('frac-thick', {'type': 'avg', 'freq': 'step'})

# Specific means-stds of dataset
means = torch.Tensor(np.array([0.03080581, 0.02708464, 0.02514398, 0.02341186, 0.02737163, 0.04289838, 0.04992491, 0.04809558, 0.05354442, 0.01810876, 0.00307337, 0.02943557, 0.0192993])).to(device)
stds = torch.Tensor(np.array([0.01119709, 0.0119086, 0.01183478, 0.01324528, 0.01333806, 0.01468573, 0.01603724, 0.01586801, 0.01681677, 0.00745883, 0.00148254, 0.01221377, 0.00994267])).to(device)

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

print("Starting training loop...")
if EVAL_ONLY:
	model.eval()
else:
	model.train()
tot_iter = 0
img_idx = 0
outer_loop_active = True
while outer_loop_active:

	curr_img_batch = torch.zeros(BATCH_SIZE, input_dim, IM_H, IM_W).to(device)
	curr_gt_batch = torch.zeros(BATCH_SIZE, IM_H, IM_W).to(device)

	for batch_ctr in range(BATCH_SIZE):
		
		# Read image and gt
		file2read = netCDF4.Dataset(SPEC_PATHS[img_idx],'r')
		layers = []
		for i, band in enumerate(BANDS):
			layers.append(file2read.variables[band][:][:, :, np.newaxis])
		img = np.concatenate(layers, axis=2)
		gt = file2read.variables['Label'][:]
		assert np.min(gt) >= 0 and np.max(gt)<= 5
		H, W, C = img.shape
	
		# Update counters
		img_idx += 1
		tot_iter += 1
		
		# Break training loop / reset image counter
		if not EVAL_ONLY and tot_iter > 100 and tot_iter % 200000 == 0:
			# Occassionally save model weights
			torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % tot_iter))
		if tot_iter >= NUM_TRAIN_ITER:
			outer_loop_active = False
			break
		elif img_idx >= len(SPEC_PATHS):
			# Re-shuffle data if reached end of epoch
			if EVAL_ONLY:
				outer_loop_active = False
				break
			perm = list(np.random.permutation(len(SPEC_PATHS)))
			SPEC_PATHS = [SPEC_PATHS[p] for p in perm]
			img_idx = 0
		
		# Setup some things
		img_torch = torch.permute((torch.Tensor(img).to(device) - means) / stds, [2, 0, 1])
		gts_torch = torch.Tensor(gt).to(device)
		gts_cloud = (gts_torch == 3) + 2 * (gts_torch == 4)  # 3 --> thin = 1, 4 --> thick = 2
		
		# The below for-loop sets -1 on those pixels which
		# we do NOT want to be part of the 'no cloud' class
		for cls_idx in CLASSES_AS_NOT_PART_OF_CLEAR:
			gts_cloud[gts_torch == cls_idx] = -1
		
		# Get a random crop
		crop_coords = get_random_crop(H, W, IM_H)[0]
		curr_gt_cloud = gts_cloud[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
		
		# Ensure crop always thin cloud (if possible), or otherwise thick cloud (if possible)
		MAX_NBR_TRYS = 51
		if torch.any(gts_cloud == 1):  # include thin cloud
			try_ctr = 0
			while not torch.any(curr_gt_cloud == 1) and try_ctr < MAX_NBR_TRYS:
				crop_coords = get_random_crop(H, W, IM_H)[0]
				curr_gt_cloud = gts_cloud[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
				try_ctr += 1
		elif torch.any(gts_cloud == 2):  # include thick cloud
			try_ctr = 0
			while not torch.any(curr_gt_cloud == 2) and try_ctr < MAX_NBR_TRYS:
				crop_coords = get_random_crop(H, W, IM_H)[0]
				curr_gt_cloud = gts_cloud[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
				try_ctr += 1
		curr_img_torch = img_torch[:, crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
		curr_img_torch_orig = curr_img_torch  # un-augmented, for visualization
		curr_gt_cloud_orig = curr_gt_cloud  # un-augmented, for visualization

		# Track ground truth stats
		sc.s('frac-img-thin').collect(torch.any(curr_gt_cloud == 1).float().cpu().detach().numpy())
		sc.s('frac-img-thick').collect(torch.any(curr_gt_cloud == 2).float().cpu().detach().numpy())
		sc.s('frac-thin').collect(torch.mean((curr_gt_cloud == 1).float()).cpu().detach().numpy())
		sc.s('frac-thick').collect(torch.mean((curr_gt_cloud == 2).float()).cpu().detach().numpy())

		# Perform data augmentation
		if (not EVAL_ONLY) and random.choice([False, True]):  # Flips image left-right with probability 50%
			curr_img_torch = torch.flip(curr_img_torch, dims=[2])
			curr_gt_cloud = torch.fliplr(curr_gt_cloud)
		if (not EVAL_ONLY) and random.choice([False, True]):  # Flips image up-down with probability 50%
			curr_img_torch = torch.flip(curr_img_torch, dims=[1])
			curr_gt_cloud = torch.flipud(curr_gt_cloud)

		# Insert to batch
		curr_img_batch[batch_ctr, :, :, :] = curr_img_torch
		curr_gt_batch[batch_ctr, :, :] = curr_gt_cloud

	# Continue to next iteration if only -1s
	if torch.all(curr_gt_batch < 0):
		continue

	# Perform prediction and compute loss
	preds = model(curr_img_batch)
	loss = criterion_CE(preds, curr_gt_batch.long())
		
	# Compute mIoU
	# See https://datascience.stackexchange.com/questions/104746/is-there-an-official-procedure-to-compute-miou
	# where it say there is no 'official' way to compute mIoU. Either one can do it by computing the mIoU per image
	# in a dataset, and then at the end just averagin the per-image-mIoUs (i.e. divide the sum of the mIoUs with 
	# the number of images in the dataset). OR one can rather keep track of a 'global' estimate across the whole
	# dataset, where one then also needs to count the occurrence of each class per image, so that a proper
	# weighted average across the whole dataset may be taken in the end. As this is more compliated, we opt here
	# for the per-image-unifrom-weighted approach. Finally, note that the difference between the two
	# approaches will diminish with increasing batch size.
	map_pred_flat = torch.argmax(preds, 1).cpu().detach().contiguous().view(-1).numpy()  # make it 1D
	gt_flat = curr_gt_batch.cpu().detach().contiguous().view(-1).numpy()  # make it 1D
	
	# Take the -1 category out of the picture
	map_pred_flat = map_pred_flat[gt_flat >= 0]
	gt_flat = gt_flat[gt_flat >= 0]
	
	# Track IoU and mIoU stats
	sc = mIoU(map_pred_flat, gt_flat, sc, nbr_classes=output_dim)

	# Track loss and other stats
	sc.s('CE_loss').collect(loss.cpu().detach().numpy())
	pred_amaxes = torch.argmax(preds, 1)
	pred_gt_check = (pred_amaxes[curr_gt_batch >= 0] == curr_gt_batch[curr_gt_batch >= 0]).float()
	sc.s('acc').collect(torch.mean(pred_gt_check).cpu().detach().numpy())
	if torch.any(curr_gt_batch == 0):
		pred_gt_check_no_cloud = (pred_amaxes[curr_gt_batch == 0] == 0).float()
		sc.s('recall-no-cloud').collect(torch.mean(pred_gt_check_no_cloud).cpu().detach().numpy())
	if torch.any(curr_gt_batch == 1):
		pred_gt_check_thin_cloud = (pred_amaxes[curr_gt_batch == 1] == 1).float()
		sc.s('recall-cloud-thin').collect(torch.mean(pred_gt_check_thin_cloud).cpu().detach().numpy())
	if torch.any(curr_gt_batch == 2):
		pred_gt_check_reg_cloud = (pred_amaxes[curr_gt_batch == 2] == 2).float()
		sc.s('recall-cloud-reg').collect(torch.mean(pred_gt_check_reg_cloud).cpu().detach().numpy())
	if torch.any(pred_amaxes == 0):
		prec = torch.count_nonzero(curr_gt_batch[pred_amaxes == 0] == 0) / torch.count_nonzero(pred_amaxes == 0)
		sc.s('precision-no-cloud').collect(prec.cpu().detach().numpy())
	if torch.any(pred_amaxes == 1):
		prec = torch.count_nonzero(curr_gt_batch[pred_amaxes == 1] == 1) / torch.count_nonzero(pred_amaxes == 1)
		sc.s('precision-cloud-thin').collect(prec.cpu().detach().numpy())
	if torch.any(pred_amaxes == 2):
		prec = torch.count_nonzero(curr_gt_batch[pred_amaxes == 2] == 2) / torch.count_nonzero(pred_amaxes == 2)
		sc.s('precision-cloud-reg').collect(prec.cpu().detach().numpy())

	# Print out how things are progressing
	sc.print()
	sc.save()
	print(tot_iter, NUM_TRAIN_ITER, img_idx, len(SPEC_PATHS))
	
	# Plot results
	if DO_PLOT:
		img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3], :]
		gt = gt[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
		gt_cloud = curr_gt_cloud_orig.cpu().detach().numpy()
		pred = model(torch.unsqueeze(curr_img_torch_orig, dim=0))
		pred_amax = torch.squeeze(torch.argmax(pred, 1)).cpu().detach().numpy()
		
		# Show RGB part of image with predictions + GT
		if SKIP_BAND_1:
			rgb_img = img[:, :, [2,1,0]] / np.max(img[:, :, [2,1,0]])
		else:
			rgb_img = img[:, :, [3,2,1]] / np.max(img[:, :, [3,2,1]])

		fig = plt.figure(figsize=(16, 16))

		if PLOT_FOR_ARTICLE:
			
			plt.subplots_adjust(wspace=0.03, hspace=0)
			
			fig.add_subplot(1,3,1)
			plt.imshow(rgb_img)
			plt.title('image')
			plt.axis('off')

			fig.add_subplot(1,3,2)
			# Note: vmax set to 3 to make it 'corresponding' with the ground truth color scale which includes undefined
			plt.imshow(pred_amax, vmin=0, vmax=3)
			plt.title('pred')#'pred-thresh, acc: %.3f, acc-cloud: %.3f,\n acc-no-cloud: %.3f, acc-reg: %.3f, acc-thin: %.3f' % (acc, acc_cloud, acc_no_cloud, acc_cloud_reg, acc_cloud_thin))
			plt.axis('off')

			fig.add_subplot(1,3,3)
			gt_cloud[gt_cloud == -1] = 3  # fully yellow = 'unconsidered' class
			plt.imshow(gt_cloud, vmin=0, vmax=3)
			plt.title('gt-cloud')
			plt.axis('off')
		else:
			fig.add_subplot(2,2,1)
			plt.imshow(rgb_img)
			plt.title('image')
			plt.axis('off')

			fig.add_subplot(2,2,2)
			plt.imshow(gt, vmin=0, vmax=5)
			plt.title('gt')
			plt.axis('off')
			
			fig.add_subplot(2,2,3)
			gt_cloud[gt_cloud == -1] = 3  # fully yellow = 'unconsidered' class
			plt.imshow(gt_cloud, vmin=0, vmax=3)
			plt.title('gt-cloud')
			plt.axis('off')

			fig.add_subplot(2,2,4)
			# Note: vmax set to 3 to make it 'corresponding' with the ground truth color scale which includes undefined
			plt.imshow(pred_amax, vmin=0, vmax=3)
			plt.title('pred')#'pred-thresh, acc: %.3f, acc-cloud: %.3f,\n acc-no-cloud: %.3f, acc-reg: %.3f, acc-thin: %.3f' % (acc, acc_cloud, acc_no_cloud, acc_cloud_reg, acc_cloud_thin))
			plt.axis('off')

		# Save figure
		plt.savefig(os.path.join(stat_train_dir, 'preds_%d.png' % (img_idx - 1)))
		plt.cla()
		plt.clf()
		plt.close('all')

	# Update model weights
	if not EVAL_ONLY:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

# After training, save model weights
if not EVAL_ONLY:
	print("Saving model weights")
	torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % NUM_TRAIN_ITER))

print("DONE! See results in logdir")
print(log_dir)
