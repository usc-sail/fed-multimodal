# Feature processing

## Table of Contents
* Applications supported
* Data partitioning
* Simulation features
* Feature processing


## Applications supported
* Speech Emotion Recognition
* Multimedia Action Recognition
* Human Activity Recognition

### Speech Emotion Recognition

Dataset | Modality | Paper | Num. of Speakers | Hours of data
|---|---|---|---|---|
MELD | A+T+V | [arxiv](https://arxiv.org/abs/1810.02508) | 260 |
MSP-Podcast | A+T(ASR) | [TAFFC'19](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Lotfian_2019_3.pdf) | >200 |


### Multimedia Action Recognition

Dataset | Modality | Paper | Num. of Instance | Hours of data
|---|---|---|---|---|
UCF101 | A+V | [arxiv](https://arxiv.org/abs/1212.0402) | 6810 |
MIT | A+V | [arxiv](https://arxiv.org/abs/1801.03150) | >100k |


## Step 1: Data partitioning
The data partioning has 2 possibilities, and it will be based on dataset:

* For many sentiment-related dataset, the client will be decided by the speaker
* For datasets that do not have natural partition (e.g. speaker), the data will be partitioned using direchlet allocation 


If we use the UCF101 dataset as an example:

To process the location data, we run the following:

```
cd data_partitioning/ufc101
python3 data_partition.py --alpha ALPHA_VALUE_0_TO_1 --raw_data_dir PATH --output_dir PATH
```

Each client's data follows the following format: 
### [key, data_file, label]


## Step 2: Simulate Federated Features
The current simulation has several features:

* Missing modality (Follow Bernoulli distribution)
* Label noise (Follow FedAudio)
* Missing labels (Follow Bernoulli distribution)


#### To simulate missing modality:

```
cd simulation_features/ufc101
python3 simulation_feature.py --alpha ALPHA_VALUE_0_TO_1 --en_missing_modality --missing_modailty_rate VALUE_0_TO_1 --output_dir PATH
```

#### To simulate label noise:

```
cd simulation_features/ufc101
python3 simulation_feature.py --alpha ALPHA_VALUE_0_TO_1 --en_label_nosiy --label_nosiy_level VALUE_0_TO_1  --output_dir PATH
```


#### To simulate missing labels:

```
cd simulation_features/ufc101
python3 simulation_feature.py --alpha ALPHA_VALUE_0_TO_1 --en_missing_label --missing_label_rate VALUE_0_TO_1  --output_dir PATH
```

Each client's data follows the following format: 
### [key, data_file, label, [MA_miss, MB_miss, label_with_noise, missing_label]]


* MA_miss = 0, Modality A is not missing
* MB_miss = 0, Modality B is not missing
* missing_label = 0, label is not missing

### The simulation feature is independ of feature processing

## Step 3: Data Features

We generate pretrained features for each data set:

* Audio - MFCC 80 dim
* Framewise Video - MobileNetv2
* Text - MobileBert

To generate features for UCF data set:

```
cd feature_processing/ufc101

# extract mobilenet_v2 feature
python3 extract_frame_feature.py --feature_type mobilenet_v2 --alpha ALPHA_VALUE_0_TO_1 --raw_data_dir PATH --output_dir PATH

# extract mfcc feature
python3 extract_audio_feature.py --feature_type mfcc --alpha ALPHA_VALUE_0_TO_1 --raw_data_dir PATH --output_dir PATH
```

Now you should be able to the output folder with the following folders:

/feature  
/partition
/simulation_feature

