#from __future__ import print_function
import os
import time
import datetime
from shutil import copyfile
import matplotlib.pyplot as plt
import torch
import xarray as xr
import numpy as np
from skimage import measure
import json
from utils import mlp_inference, MLP5, eval_swe_forest_cls

BASE_PATH_LOG = '../log'
BASE_PATH_DATA = '../data/skogsstyrelsen/'
MODEL_LOAD_PATH = None
MODEL_LOAD_PATH = ['../log/2023-08-10_11-49-01/model_it_2000000', '../log/2023-08-10_11-49-22/model_it_2000000', '../log/2023-08-10_11-49-49/model_it_2000000',
				   '../log/2023-08-10_11-50-44/model_it_2000000', '../log/2023-08-10_11-51-11/model_it_2000000', '../log/2023-08-10_11-51-36/model_it_2000000',
				   '../log/2023-08-10_11-51-49/model_it_2000000', '../log/2023-08-10_11-52-02/model_it_2000000', '../log/2023-08-10_11-52-24/model_it_2000000',
				   '../log/2023-08-10_11-52-47/model_it_2000000']
SPLIT_TO_USE = 'trainval'  # 'train', 'val', 'trainval' or 'test'
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DO_PLOT = False
SHOW_CLOUD_CONTOUR_ON_IMG = True
THRESHOLD_THICKNESS_IS_CLOUD = 0.010  # if COT predicted above this, then predicted as 'opaque cloud' ("thick" cloud)
THRESHOLD_THICKNESS_IS_THIN_CLOUD = 0.010  # if COT predicted above this, then predicted as 'thin cloud' <-- set to the same as the opaque cloud threshold by default, i.e. it becomes a binary task (cloudy / clear) instead
SKIP_BAND_10 = True  # True --> Skip band 10 as input (may be needed for Skogsstyrelsen data)
SKIP_BAND_1 = True  # True --> Skip band 1 as input (typicall done if trained on SMHI)
BAND_NAMES = ['b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b8a', 'b09', 'b10', 'b11', 'b12']
MLP_POST_FILTER_SZ = 2  # 1 --> no filtering, >= 2 --> majority vote within that-sized square
PRED_BASED_ON_SCL_LAYER = False  # True --> can try out how well the ESA SCL layer fares on this task!
SCL_COLORS = {0: np.array([0, 0, 0]), # No Data (black)
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

if not isinstance(MODEL_LOAD_PATH, list):
	MODEL_LOAD_PATH = [MODEL_LOAD_PATH]
if not isinstance(THRESHOLD_THICKNESS_IS_CLOUD, list):
	THRESHOLD_THICKNESS_IS_CLOUD = [THRESHOLD_THICKNESS_IS_CLOUD]
if not isinstance(THRESHOLD_THICKNESS_IS_THIN_CLOUD, list):
	THRESHOLD_THICKNESS_IS_THIN_CLOUD = [THRESHOLD_THICKNESS_IS_THIN_CLOUD]

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)
copyfile("swe_forest_agency_cls.py", os.path.join(log_dir, "swe_forest_agency_cls.py"))

def color_scl_correctly(scl_layer):
	# Function for setting the correct colors so that they match
	# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
	scl_layer_3chan = np.zeros((scl_layer.shape[0], scl_layer.shape[1], 3), dtype=int)
	for key, value in SCL_COLORS.items():
		if key > 0:  # <-- not needed for key=0 since scl_layer_3chan initialized as zeros
			scl_layer_3chan[scl_layer == key, :] = value
	return scl_layer_3chan

# Read data + corresponding json info (incl ground truth)
img_paths_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_train.npy')))
img_paths_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_val.npy')))
img_paths_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_test.npy')))
json_content_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_train.npy'), allow_pickle=True))
json_content_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_val.npy'), allow_pickle=True))
json_content_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_test.npy'), allow_pickle=True))

# Specify model input and output dimensions
input_dim = 13 - SKIP_BAND_10 - SKIP_BAND_1
output_dim = 1

# Load means and stds based on the synthetic data
means = torch.Tensor(np.array([0.64984976, 0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])).to(DEVICE)
stds = torch.Tensor(np.array([0.3596485, 0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])).to(DEVICE)

if SKIP_BAND_10:
	means = means[[0,1,2,3,4,5,6,7,8,9,11,12]]
	stds = stds[[0,1,2,3,4,5,6,7,8,9,11,12]]
	BAND_NAMES.remove('b10')
if SKIP_BAND_1:
	means = means[1:]
	stds = stds[1:]
	BAND_NAMES.remove('b01')

# Setup and load model
models = []
for model_load_path in MODEL_LOAD_PATH:
	model = MLP5(input_dim, output_dim, apply_relu=True)
	model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
	model.to(DEVICE)
	models.append(model)

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
all_binary_preds = []
all_binary_gts = []
for img_idx, img_path in enumerate(img_paths):

	print(img_idx, len(img_paths))

	# Extract date to see if data is from before or after Jan 2022
	# (this affects the normalization used for the image)
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

	# Extract image shape
	H, W = img.shape[:2]

	# read GT (molndis = 1 --> cloudy, molndis = 0 --> clear)
	molndis = json_paths[img_idx]['MolnDis']
	
	# Perform prediction
	pred_map, pred_map_binary_list, pred_map_binary_thin_list = mlp_inference(img, means, stds, models, H*W,
																			  THRESHOLD_THICKNESS_IS_CLOUD,
																			  THRESHOLD_THICKNESS_IS_THIN_CLOUD,
																			  MLP_POST_FILTER_SZ, DEVICE)

	# Track stats
	pred_map_binary = pred_map_binary_list[0]
	pred_map_binary_thin = pred_map_binary_thin_list[0]
	frac_binary = 100*np.count_nonzero(pred_map_binary + pred_map_binary_thin) / H / W
	if PRED_BASED_ON_SCL_LAYER:
		pred_cloudy = np.any(np.logical_and(scl_layer_raw >= 8, scl_layer_raw <= 10))
	else:
		pred_cloudy = frac_binary > 0  # any pixel(s) cloudy --> predicted as cloudy
	all_binary_preds.append(int(pred_cloudy))
	all_binary_gts.append(int(molndis))

	# Visualize results
	if DO_PLOT:
		min_img_rgb = np.array([2.00000009e-03, 3.00000014e-04, 9.99999975e-05])
		max_img_rgb = np.array([0.912, 0.90399998, 0.9016])
		nbr_rows = 1
		fig = plt.figure(figsize=(16, 16))
		if SKIP_BAND_1:
			rgb_img = (img[:, :, [2,1,0]] - min_img_rgb) / max_img_rgb
		else:
			rgb_img = (img[:, :, [3,2,1]] - np.nanmin(img[:, :, [3,2,1]])) / np.nanmax(img[:, :, [3,2,1]])
		fig.add_subplot(nbr_rows,4,1)
		plt.imshow(rgb_img, vmin=0, vmax=1)
		if SHOW_CLOUD_CONTOUR_ON_IMG:
			contours = measure.find_contours(0.0 + pred_map_binary + pred_map_binary_thin, 0.9)
			for contours_entry in contours:
				plt.plot(contours_entry[:, 1], contours_entry[:, 0], color='r')
		plt.title('image')
		plt.axis('off')
		
		fig.add_subplot(nbr_rows,4,2)
		plt.title('pred (min, max)=(%.3f, %.3f)' % (np.nanmin(pred_map), np.nanmax(pred_map)))
		pred_map[np.isnan(pred_map)] = 0
		plt.imshow(pred_map, vmin=0, vmax=1, cmap='gray')
		plt.axis('off')

		fig.add_subplot(nbr_rows,4,3)
		plt.imshow(0.0 + 2*pred_map_binary + pred_map_binary_thin, vmin=0, vmax=2, cmap='gray')
		if pred_cloudy:
			plt.title('pred-binary, cloudy (%.1f prct) | gt: %s' % (frac_binary, molndis))
		else:
			plt.title('pred-binary, OK (%.1f prct) | gt: %s' % (frac_binary, molndis))
		plt.axis('off')
		
		# Show also SCL band
		fig.add_subplot(nbr_rows,4,4)
		plt.imshow(scl_layer, vmin=0, vmax=255)
		plt.title('scl')
		plt.axis('off')

		plt.savefig(os.path.join(stat_train_dir, '../skogsstyrelsen_%d.png' % (img_idx)))
		plt.cla()
		plt.clf()
		plt.close('all')
		print("PLOTTED", img_idx)

# Save predictions
all_binary_preds = np.array(all_binary_preds)
np.save('skogs_preds.npy', all_binary_preds)

# Evaluate predictions
eval_swe_forest_cls(BASE_PATH_DATA, SPLIT_TO_USE)

print("DONE")
