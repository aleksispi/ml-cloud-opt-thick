#from __future__ import print_function
import os
import sys
import time
import random
import datetime
from shutil import copyfile
import torch
import torch.nn as nn
import numpy as np
from utils import StatCollector, MLP5


# Global vars
BASE_PATH_DATA = '../data/synthetic-cot-data'
BASE_PATH_LOG = '../log'
USE_GPU = False  # The code uses small-scale models, GPU doesn't seem to accelerate things actually
SEED = 0
SPLIT_TO_USE = 'train'  # 'train', 'val' or 'test'
BATCH_SIZE = 32
INPUT_NOISE = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]  # Add 0-mean white noise with std being a fraction of the mean input of train, to data inputs
OUTPUT_DO_DENORMALIZE = True  # True --> errors reported in same "scale" as the regressor's original values
SKIP_BAND_10 = False  # True --> Skip band 10 as input (may be needed for Skogsstyrelsen data)
SKIP_BAND_1 = True  # True --> Skip band 1 (SKIP_BAND_1 should always be True, as the band currently does not make sense in the data; further work needed in that direction in future work)
PROPERTY_COLUMN_MAPPING = {'spec_bands': [i for i in range(1 + SKIP_BAND_1, 14)], 'angles': [14, 15, 16], 'thick': [17], 'type': [18], 'prof_id': [19], 'gas_vapour': [20, 21], 'surf_prof': [22]}
INPUTS = ['spec_bands']  # Add keys from PROPERTY_COLUMN_MAPPING to use those as inputs
REGRESSOR = 'thick'  # Can be set to any other key in PROPERTY_COLUMN_MAPPING to regress that instead
THRESHOLD_THICKNESS_IS_CLOUD = 0.025  # Cloud optical tickness (COT) above this --> seen as an 'opaque cloud' pixel
THRESHOLD_THICKNESS_IS_THIN_CLOUD = 0.015  # Cloud optical tickness (COT) above this --> seen as a 'thin cloud' pixel
MODEL_LOAD_PATH = None  # See examples on how to point to a specific model path below. None --> Model randomly initialized.

# Note: Some of the things evaluated (but where we have moved on and not saved the result folders), includes:
# - different types of normalization (min-max, min-max-avoid-outlier --> both worse than standard mean-std)
# - vary beta1 in Adam --> 0.9 was best
# - various learning rates --> 0.0003 was best
# - various batch sizes --> 32 was best
# - dropout --> no dropout was best
# - ELU and LeakyReLU --> worse than ReLU
# - residual connections in NN model --> does not improve results
# - L1 and Huber loss instead of L2 --> no effect
# - 2-way models (can predict cloud type in addition to thickness), but we found using only thickness predictors was best
# - combine kappa and smhi data in various ways, no good results

# 10 models trained on SMHI synthetic data, with 12 bands, 3% additive noise
MODEL_LOAD_PATH = ['../log/2023-08-10_10-33-44/model_it_2000000', '../log/2023-08-10_10-34-06/model_it_2000000', '../log/2023-08-10_10-34-18/model_it_2000000',
				   '../log/2023-08-10_10-34-28/model_it_2000000', '../log/2023-08-10_10-34-46/model_it_2000000', '../log/2023-08-10_10-34-58/model_it_2000000',
				   '../log/2023-08-10_10-35-09/model_it_2000000', '../log/2023-08-10_10-35-31/model_it_2000000', '../log/2023-08-10_10-35-52/model_it_2000000',
				   '../log/2023-08-10_10-36-12/model_it_2000000']

# As above, but omit band 10 also (for use in skogs data)
#MODEL_LOAD_PATH = ['../log/2023-08-10_11-49-01/model_it_2000000', '../log/2023-08-10_11-49-22/model_it_2000000', '../log/2023-08-10_11-49-49/model_it_2000000',
#				   '../log/2023-08-10_11-50-44/model_it_2000000', '../log/2023-08-10_11-51-11/model_it_2000000', '../log/2023-08-10_11-51-36/model_it_2000000',
#				   '../log/2023-08-10_11-51-49/model_it_2000000', '../log/2023-08-10_11-52-02/model_it_2000000', '../log/2023-08-10_11-52-24/model_it_2000000',
#				   '../log/2023-08-10_11-52-47/model_it_2000000']


# MODEL_LOAD_PATH must be a list of model paths
if not isinstance(MODEL_LOAD_PATH, list):
	MODEL_LOAD_PATH = [MODEL_LOAD_PATH]
# Same for INPUT_NOISE
if not isinstance(INPUT_NOISE, list):
	INPUT_NOISE = [INPUT_NOISE]

# Specify model input and output dimensions
input_dim = np.sum([len(PROPERTY_COLUMN_MAPPING[inp]) for inp in INPUTS]) - SKIP_BAND_10
output_dim = 1

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)
copyfile("cot_synth_eval.py", os.path.join(log_dir, "cot_synth_eval.py"))

# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(SEED)

# Read data
trainset = np.load(os.path.join(BASE_PATH_DATA, 'trainset_smhi.npy'))
valset = np.load(os.path.join(BASE_PATH_DATA, 'valset_smhi.npy'))
testset = np.load(os.path.join(BASE_PATH_DATA, 'testset_smhi.npy'))
nbr_examples_train = trainset.shape[0]
nbr_examples_val = valset.shape[0]
nbr_examples_test = testset.shape[0]
nbr_examples = nbr_examples_train + nbr_examples_val + nbr_examples_test
print("Done reading data")

# Separate input and regression variable
inputs_train = np.concatenate([trainset[:, PROPERTY_COLUMN_MAPPING[inp]] for inp in INPUTS], axis=1)
gts_train = np.squeeze(trainset[:, PROPERTY_COLUMN_MAPPING[REGRESSOR]])
gts_train_binary = np.squeeze(trainset[:, PROPERTY_COLUMN_MAPPING['type']]) > 0
inputs_val = np.concatenate([valset[:, PROPERTY_COLUMN_MAPPING[inp]] for inp in INPUTS], axis=1)
gts_val = np.squeeze(valset[:, PROPERTY_COLUMN_MAPPING[REGRESSOR]])
gts_val_binary = np.squeeze(valset[:, PROPERTY_COLUMN_MAPPING['type']]) > 0
inputs_test = np.concatenate([testset[:, PROPERTY_COLUMN_MAPPING[inp]] for inp in INPUTS], axis=1)
gts_test = np.squeeze(testset[:, PROPERTY_COLUMN_MAPPING[REGRESSOR]])
gts_test_binary = np.squeeze(testset[:, PROPERTY_COLUMN_MAPPING['type']]) > 0
if SKIP_BAND_10:
	if SKIP_BAND_1:
		inputs_train = inputs_train[:, [0,1,2,3,4,5,6,7,8,10,11]]
		inputs_val = inputs_val[:, [0,1,2,3,4,5,6,7,8,10,11]]
		inputs_test = inputs_test[:, [0,1,2,3,4,5,6,7,8,10,11]]
	else:
		inputs_train = inputs_train[:, [0,1,2,3,4,5,6,7,8,9,11,12]]
		inputs_val = inputs_val[:, [0,1,2,3,4,5,6,7,8,9,11,12]]
		inputs_test = inputs_test[:, [0,1,2,3,4,5,6,7,8,9,11,12]]

# Normalize regressor data and convert to Torch
gt_max = max(np.max(gts_train), np.max(gts_val), np.max(gts_test))
gts_test /= gt_max
gts_train /= gt_max
gts_val /= gt_max
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
gts_train = torch.Tensor(gts_train).to(device)
gts_val = torch.Tensor(gts_val).to(device)
gts_test = torch.Tensor(gts_test).to(device)

# Setup prediction model and loss
models = []
for model_load_path in MODEL_LOAD_PATH:  # Ensemble approach if len(MODEL_LOAD_PATH) > 1
	model = MLP5(input_dim, output_dim, apply_relu=True)
	if model_load_path is not None:
		model.load_state_dict(torch.load(model_load_path, map_location=device))
	model.to(device)
	model.eval()
	models.append(model)
criterion = nn.L1Loss().to(device)  # NB: L1Loss used in eval to get RMSE instead (no backprop, just eval)

# Setup statistics collector
sc = StatCollector(stat_train_dir, 9999999, 10)
sc_loss_string_base = 'MSE_loss'
sc.register(sc_loss_string_base + '_' + SPLIT_TO_USE, {'type': 'avg', 'freq': 'step'})
sc.register('RMSE_' + SPLIT_TO_USE, {'type': 'avg', 'freq': 'step'})

if SPLIT_TO_USE == 'train':
	inputs = inputs_train
elif SPLIT_TO_USE == 'val':
	inputs = inputs_val
elif SPLIT_TO_USE == 'test':
	inputs = inputs_test
inputs_train_orig = inputs_train
inputs_orig = inputs
for noise_idx, noise in enumerate(INPUT_NOISE):
	
	# Add noise disturbances to data (if enabled)
	means_input_train = np.mean(inputs_train_orig, axis=0)
	white_noise = np.random.randn(*inputs_orig.shape) * means_input_train[np.newaxis, :] * noise
	inputs = inputs_orig + white_noise
	
	# Normalize input data
	means_input_train = np.mean(inputs_train_orig, axis=0)
	stds_input_train = np.std(inputs_train_orig, axis=0)
	inputs = (inputs - means_input_train) / stds_input_train

	# Convert to Torch
	inputs = torch.Tensor(inputs).to(device)

	# Extract things based on split
	if SPLIT_TO_USE == 'train':
		nbr_examples = nbr_examples_train
		gts = gts_train
	elif SPLIT_TO_USE == 'val':
		nbr_examples = nbr_examples_val
		gts = gts_val
	elif SPLIT_TO_USE == 'test':
		nbr_examples = nbr_examples_test
		gts = gts_test
		
	# Run evaluation
	print("Running evaluation (%s set)..." % SPLIT_TO_USE)
	it = 0
	while True:
		
		# Break evaluation loop at end of epoch
		if (it + 1) * BATCH_SIZE >= nbr_examples:
			break
		
		# Compute a prediction and get the loss
		curr_gts = gts[it * BATCH_SIZE : (it + 1) * BATCH_SIZE]
		preds = 0
		for model in models:
			curr_pred = model(inputs[it * BATCH_SIZE : (it + 1) * BATCH_SIZE, :]) / len(models)
			preds += curr_pred
		loss = criterion(preds[:, 0], curr_gts)
		loss_to_sc = loss.cpu().detach().numpy()
		sc.s(sc_loss_string_base + '_' + SPLIT_TO_USE).collect(loss_to_sc)
		if OUTPUT_DO_DENORMALIZE:
			loss_to_sc *= gt_max  # RMSE "denormalized"
		sc.s('RMSE_' + SPLIT_TO_USE).collect(loss_to_sc)

		# Track statistics
		if it % 100 == 0:
			sc.print()
			sc.save()
			print("Iter: %d / %d" % (it, nbr_examples // BATCH_SIZE))
		it += 1
	print("Done with %s set evaluation!" % SPLIT_TO_USE)

print("DONE! Final stats (see 'tot' -- forget 'ma' and 'last' below):")
sc.print()
