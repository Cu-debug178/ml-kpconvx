# KPConvX: Standalone code

This folder contains a standalone version of KPConvX code.

## Setup

Our code was tested with multiple environments and should be straightforward to setup. Addapt the following lines to your environment and version of CUDA:

```bash
conda create -n kpconvx python=3.10
conda activate kpconvx
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install easydict h5py matplotlib numpy scikit-learn timm pykeops
pip install 'pyvista[all,trame]' jupyterlab
```


## Prepare data

### S3DIS

We use the preprocessed data from Pointcept that you can download [here](https://huggingface.co/datasets/Pointcept/s3dis-compressed). Please agree with the official license before downloading it. From the `s3dis.tar.gz` archive, extract the `s3dis` folder to the `Standalone/data` directory.
 
### ScanObjectNN

We use the h5_files from [official website](https://hkust-vgd.github.io/scanobjectnn/). Download the preprocessed version and place it in the `Standalone/data` directory. Rename the folder `h5_files` to `ScanObjectNN`. You should have (and only need) the three following files:
```
data/ScanObjectNN/main_split/test_objectdataset_augmentedrot_scale75_1024_fps.pkl
data/ScanObjectNN/main_split/test_objectdataset_augmentedrot_scale75.h5
data/ScanObjectNN/main_split/training_objectdataset_augmentedrot_scale75.h5
```

## Experiments


### Training networks

We provide scripts to train our model on ScanObjectNN or S3DIS.

```bash
# Training on ScanObjectNN
./train_ScanObjectNN.sh

# Training on S3DIS
./train_S3DIS.sh
```

Follow instructions in these scripts to change parameters.


### Plotting functions

We provide plotting functions to plot the performances during and after training. They are in the `plot_ScanObj.py` and `plot_S3DIS.py` script. Here are 

- Step 1: Define the dates of the logs you want to plot in the experiment functions. See example functions `experiment_name_1()` and `experiment_name_2()`
```python
def experiment_name_1():
    ...
    start = 'Log_2020-04-22_11-52-58'
    end = 'Log_2023-07-29_12-40-27'
    ...
```

- Step 2: Choose the log to show 
```python
# Choose the logs to show
logs, logs_names = experiment_name_1()
```

- Step 3: Run the script
```bash
python3 experiments/ScanObjectNN/plot_ScanObj.py
# or
python3 experiments/S3DIS/plot_S3DIS.py
```

Once the trainings are finished, you can change to test mode with `perform_test = True`. This will start tests for the selected trained weights and show a summary of the test results.


### Test models

You can also directly test a trained network with the following scripts.

```bash
# Test a model on ScanObjectNN
./test_ScanObjectNN.sh

# Test a model on S3DIS
./test_S3DIS.sh
```

More detailed instructions are in these scripts.


### Pretrained weights

We provide the following pretrained models:

| Model | Benchmark | OA | mAcc | Size | Archive |
| :---: | :---: | :---: | :---: | :---: | :---: |
| KPConvD-L | ScanObjectNN   | 89.7% | 88.5% | 80 MB | [link](https://ml-site.cdn-apple.com/models/kpconvx/ScanObjectNN_KPConvD-L.zip) |
| KPConvX-L | ScanObjectNN   | 89.1% | 87.6% | 138 MB | [link](https://ml-site.cdn-apple.com/models/kpconvx/ScanObjectNN_KPConvX-L.zip) |

| Model | Benchmark | Val mIoU | Size | Archive |
| :---: | :---: | :---: | :---: | :---: |
| KPConvD-L | S3DIS (Area5)  | 72.3% | 151 MB | [link](https://ml-site.cdn-apple.com/models/kpconvx/S3DIS_KPConvD-L.zip) |
| KPConvX-L | S3DIS (Area5)  | 73.5% | 169 MB | [link](https://ml-site.cdn-apple.com/models/kpconvx/S3DIS_KPConvX-L.zip) |

You can download and extract them to the result folder and use our scripts to test them.



## FastAdapter-inspired KPConvX

The S3DIS and ScanObjectNN configurations include an optional sampler-agnostic
P2A/A2P context path. It keeps the existing KPConvX hierarchy intact: anchors
are selected once per cloud, then used for geometry-aware P2A aggregation and
A2P feature compensation at every encoder stage.

```bash
./train_S3DIS_fastadapter.sh
./train_ScanObjectNN_fastadapter.sh
```

Use `--fa_enabled 0` for the baseline. The anchor source can be selected with
`--fa_anchor_mode fps|pyramid|random|stride`; `pyramid` additionally accepts
`--fa_anchor_level`. For a checkpoint-backed Adapter/head-only run, pass
`--fa_train_mode adapter_head --finetune_path /path/to/checkpoint.tar`.














