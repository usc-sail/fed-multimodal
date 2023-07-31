
### How to run CREMA-D Dataset in FedMultimodal (Frame-wise Video and Audio)
Here we provide an example to quickly start with the experiments, and reproduce the CREMA-D results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.

#### 0. Download data: The data will be under data/crema-d by default. Make sure you have git-lfs installed.

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_cremad.sh
cd ..
```

Data will be under data/creama_d

#### 1. Partition the data

This data has a natural partition (Speaker ID).

```
python3 features/data_partitioning/crema_d/data_partition.py
```

The return data is a list, each item containing [key, file_name, label, speaker_id], e.g.:

```
[
    "1020/1020_IEO_NEU_XX",
    "/home/tiantiaf/fed-multimodal/fed_multimodal/data/crema_d/CREMA-D/AudioWAV/1020_IEO_NEU_XX.wav",
    3,
    1020
]
```

In the above example, corresponding video file is: /home/tiantiaf/fed-multimodal/fed_multimodal/data/crema_d/CREMA-D/VideoFlash/1020_IEO_NEU_XX.flv, the post processing will automatically taking care of this.

#### 2. Feature extraction

For Crema-D dataset, the feature extraction includes visual/audio feature extraction.

```
# extract mobilenet_v2 framw-wise feature
python3 features/feature_processing/crema_d/extract_frame_feature.py --feature_type mobilenet_v2

# extract mfcc (audio) feature
taskset -c 1-30 python3 features/feature_processing/crema_d/extract_audio_feature.py --feature_type mfcc
```

#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/crema_d
# output/mm/crema_d/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/crema_d
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality (the same for missing labels and label noises)
```
cd experiment/crema_d
bash run_mm.sh
```

#### Baseline results for executing the above
Dataset | Modality | Link | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | UAR (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
Crema-D | Video/Audio | [Crema-D-Data](https://github.com/CheyneyComputerScience/CREMA-D) | 4 | ~70 per fold | Natural | - |  FedAvg <br> FedOpt | 61.66% <br> 62.66% | 0.05 | 200 |

