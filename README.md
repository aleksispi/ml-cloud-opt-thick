# Creating and Leveraging a Synthetic Dataset of Cloud Optical Thickness Measures for Cloud Detection in MSI
![thumb](https://github.com/aleksispi/init-des/assets/32370520/f7a0cf68-b85c-415a-8800-7596bd996a22)

Official code and dataset repository for the Remote Sensing 2024 journal paper [_Creating and Leveraging a Synthetic Dataset of Cloud Optical Thickness Measures for Cloud Detection in MSI_](https://www.mdpi.com/2072-4292/16/4/694). Also presented as a poster at EUMETSAT 2023.

[Journal paper](https://www.mdpi.com/2072-4292/16/4/694) | [arXiv](https://arxiv.org/abs/2311.14024)

## Datasets
In this work, two novel datasets are introduced (see [our paper](https://www.mdpi.com/2072-4292/16/4/694) for details):
* A synthetic dataset for cloud optical thickness estimation, which can be downloaded [here](https://drive.google.com/drive/folders/16VBNSgT-ngsoH_ZZsDbOPbwpSB100k-1?usp=sharing).
* A dataset of real satellite images, each of which is labeled 'clear' or 'cloudy'. This dataset can be downloaded [here](https://drive.google.com/drive/folders/1lRCIcQo9CqFRDhUd3aZRAA46k8nLL49J?usp=sharing).

## If you want to set up your own Conda environment for this code base
On a Ubuntu work station, the below should be sufficient for running the code in this repository.
```
conda create -n cot_env python=3.8
conda activate cot_env
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scipy
pip install matplotlib
pip install xarray
pip install scikit-image
pip install netCDF4
```

This will create the `cot_env` environment which you will then activate with
```conda activate cot_env```

## Training and evaluating ML models for synthetic cloud optical thickness (COT) data provided by SMHI
The main files of importance are `cot_synth_train.py` and `cot_synth_eval.py`, where the workflow is oriented around first training (and at the end of training, saving) models using `cot_synth_train.py`, followed by evaluting said models using `cot_synth_eval.py`.

Begin by creating a folder `../data`, and in this folder you should put the data folder `synthetic-cot-data` that you can download [here](https://drive.google.com/drive/folders/16VBNSgT-ngsoH_ZZsDbOPbwpSB100k-1?usp=sharing) (you must also unzip the file). Then also create a folder `../log` (this folder should be side-by-side with the folder `../data`; not one inside the other).

After the above, to train a model simply run
```
python cot_synth_train.py
```
Model weights will then be automatically saved in a log-folder, e.g. `../log/data-stamp-of-folder/model_it_xxx`. Note that, by default, `cot_synth_train.py` trains models on the training set,
where each input data point is assumed to be a 12-dimensional vector corresponding to the 12 spectral bands in the synthetic dataset (all of the 13 standard bands, except for B1), and where 3% noise
is added to the inputs during training (see the flag `INPUT_NOISE_TRAIN`). To train models which also omit band B10 (e.g. to do evaluations on Swedish Forest Agency data;
see "Evaluating trained COT estimation model on use-case by the Swedish Forest Agency" below), set `SKIP_BAND_10` to True instead of False.

To evaluate model(s), update the flag `MODEL_LOAD_PATH` in `cot_synth_eval.py` so that it points to the model checkpoint file induced by running the above training command. After that, simply run
```
python cot_synth_eval.py
```
in order to evaluate the model that you trained using `cot_synth_train.py`. Note: By default, the evaluation occurs for the training split (can be changed with the flag `SPLIT_TO_USE`). Also,
by default, the flag `INPUT_NOISE` is set as the list `[0.00, 0.01, 0.02, 0.03, 0.04, 0.05]`. In this case, the evaluation script will show average results across different input noise levels
(typically results get better at lower noise levels compared to higher levels).

It is possible to evaluate ensemble models as well. To do this for an ensmble of N models, first train N models by running the `cot_synth_train.py` script N times (using different `SEED` each time
so that the models do not become identical). Then, when running `cot_synth_eval.py`, ensure MODEL_LOAD_PATH becomes a list where each list element is a model log path. Examples of this type of
path specification are already available within `cot_synth_eval.py`.

Pretrained model weights are already available [here](https://drive.google.com/drive/folders/1MkqcoxLBb9C1vAUwvHipq5cr6Z7bXIel?usp=sharing). Download the contents of this folder (should be 10 folders
in total), unzip it, and ensure the resulting 10 folders land inside `../log/`. The model weights that are available are 10 five-layer MLPs that were trained using 3% additive noise, and where
each input data point is assumed to be a 12-dimensional vector corresponding to the 12 spectral bands in the synthetic dataset (all of the 13 standard bands, except for B1).
They can be run in ensemble-mode as described in the previous paragraph.

## Evaluating trained COT estimation model on use-case by the Swedish Forest Agency (SFA, Skogsstyrelsen)
The main file of importance is `swe_forest_agency_cls.py`.

If you haven't already, begin by creating a folder `../data`, and in this folder you should create a folder `skogsstyrelsen`. Within `../data/skogsstyrelsen/`, put the data that you can download from [here](https://drive.google.com/drive/folders/1lRCIcQo9CqFRDhUd3aZRAA46k8nLL49J?usp=sharing). Then, if you haven't already, also create a folder `../log` (i.e. the `data` and `log` folders should be next to
each other; not one of them within the other).

To evaluate model(s) on the SFA cloudy / clear image binary classification setup, first ensure that `MODEL_LOAD_PATH` points to model / models that have been trained on the synthetic
data by SMHI (see "Training and evaluating ML models for synthetic cloud optical thickness (COT) data provided by SMHI" above), AND/OR first download pretrained models as described below. Then run
```
python swe_forest_agency_cls.py
```
The above will by default run the model on the whole train-val set in the provided train-val-test split. You can change what split to run the model on using the flag `SPLIT_TO_USE` in
`swe_forest_agency_cls.py`. Various results such as F1-scores will be shown after running the code.

Pretrained model weights are already available [here](https://drive.google.com/drive/folders/14xTbLHPxaPznemG7ShE0DMC9zJsNU_hr?usp=sharing). Download the folder, unzip it, and ensure the resulting
10 folders land inside `../log/`. The model weights that are available are 10 five-layer MLPs that were trained using 3% additive noise, and where each input data point is assumed to be an
11-dimensional vector corresponding to 11 out of the 12 spectral bands in the synthetic dataset (all of the 13 standard bands, except for B1 and B10). They can be run in ensemble-mode, as has been explained previously.

### ResNet-18-based image classification alternative
As described in [our paper](https://www.mdpi.com/2072-4292/16/4/694), we also compare our COT estimation-based approach with an image classification-based approach on the SFA dataset. To train a ResNet-18-based such classifier, simply run
```
python binary_cls_skogs.py
```
Once the model has finished training, ensure that `MODEL_LOAD_PATH` points to the saved model weights and set `EVAL_ONLY = True` within `binary_cls_skogs.py` in order to evaluate the trained model. The split on which the model is evaluated can be changed using the flag `SPLIT_TO_USE`.

## Training and/or evaluating on KappaZeta data
The KappaZeta dataset we have used in our paper was downloaded from [here](https://zenodo.org/records/5095024). To train FCN-based models for cloud type segmentation on KappaZeta, see the file `kappa_cloud_train.py`. In particular, a model is trained via the command 
```
python kappa_cloud_train.py
```
Note that the model is trained on data corresponding to the months April, May and June (see the flag `MONTHS`). To evaluate a trained model, ensure that `MODEL_LOAD_PATH` points to the saved model weights and set `EVAL_ONLY = True` within `kappa_cloud_train.py`. In the paper, the models are evaluated on data corresponding to the months July, August and September (thus change `MONTHS` accordingly).

To instead train / refine / evaluate MLP-based models on KappaZeta, please refer instead to the file `kappa_cloud_opt_thick.py`, which in many ways works in the same way as `kappa_cloud_train.py`.

## Citation
If you find our dataset(s), code, and/or [our paper](https://www.mdpi.com/2072-4292/16/4/694) interesting or helpful, please consider citing:

    @article{pirinen2024creating,
      title={Creating and Leveraging a Synthetic Dataset of Cloud Optical Thickness Measures for Cloud Detection in MSI},
      author={Pirinen, Aleksis and Abid, Nosheen and Paszkowsky, Nuria Agues and Timoudas, Thomas Ohlson and Scheirer, Ronald and Ceccobello, Chiara and Kov{\'a}cs, Gy{\"o}rgy and Persson, Anders},
      journal={Remote Sensing},
      volume={16},
      number={4},
      pages={694},
      year={2024},
      publisher={MDPI}
    }

## Acknowledgements
This work was funded by [Vinnova](https://www.vinnova.se/en/) ([Swedish Space Data Lab 2.0](https://www.vinnova.se/en/p/swedish-space-data-lab-2.0/), grant number 2021-03643), the [Swedish National Space Agency](https://www.rymdstyrelsen.se/en/) and the [Swedish Forest Agency](https://www.skogsstyrelsen.se/).
