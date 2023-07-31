
# FedMultimodal - 2023 KDD ADS
#### FedMutimodal [[Paper Link](https://arxiv.org/pdf/2306.09486.pdf)] is an open source project for researchers exploring multimodal applications in Federated Learning setup. FedMultimodal was accepted to 2023 KDD ADS track. 

The framework figure:

<div align="center">
 <img src="fed_multimodal/img/FedMultimodal.jpg" width="750px">
</div>

#### Image credit: https://openmoji.org/


## Applications supported
* #### Cross-Device Applications
    * Emotion Recognition [[CREMA-D](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/crema_d)] [[Meld](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/meld)]
    * Multimedia Action Recognition [[UCF-101](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/ucf101)] [[MiT-51](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/mit51)]
    * Human Activity Recognition [[UCI-HAR](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/uci-har)] [[KU-HAR](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/ku-har)] 
    * Social Media [[Crisis-MMD](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/crisis-mmd)] [[Hateful-Memes](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/hateful_memes)]
* #### Cross-silo Applications (e.g. Medical Settings)
    * ECG classification [[PTB-XL](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/experiment/ptb-xl)]
    * Ego-4D (To Appear)
    * Medical Imaging (To Appear)

### Installation
To begin with, please clone this repo:
```
git clone git@github.com:usc-sail/fed-multimodal.git
```

To install the conda environment:
```
cd fed-multimodal
conda create --name fed-multimodal python=3.9
conda activate fed-multimodal
```

Then pip install the package:
```
pip install -e .
```

### [Data processing recipe](https://github.com/usc-sail/fed-multimodal/tree/main/fed_multimodal/features)

Feature processing includes 3 steps:

* Data partitioning
* Simulation features
* Feature processing

### Quick Start -- UCI-HAR Example (Acc. and Gyro)
Here we provide an example to quickly start with the experiments, and reproduce the UCI-HAR results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.


#### 0. Download data: The data will be under data/uci-har by default. 

You can modify the data path in system.cfg to the desired path.

```
cd fed_multimodal/data
bash download_uci_har.sh
cd ..
```

#### 1. Partition the data

alpha specifies the non-iidness of the partition, the lower, the higher data heterogeneity. As each subject performs the same amount activities, we partition each subject data into 5 sub-clients.

```
python3 features/data_partitioning/uci-har/data_partition.py --alpha 0.1 --num_clients 5
python3 features/data_partitioning/uci-har/data_partition.py --alpha 5.0 --num_clients 5
```

The return data is a list, each item containing [key, file_name, label]

#### 2. Feature extraction

For UCI-HAR dataset, the feature extraction mainly handles normalization.

```
python3 features/feature_processing/uci-har/extract_feature.py --alpha 0.1
python3 features/feature_processing/uci-har/extract_feature.py --alpha 5.0
```


#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/uci-har
# output/mm/ucihar/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/uci-har
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality
```
cd experiment/uci-har
bash run_mm.sh
```

#### Baseline results for executing the above
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | F1 (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
UCI-HAR | Acc+Gyro | [UCI-Data](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 6 | 105 | Natural+Manual | 5.0 <br> 5.0 <br> 0.1 <br> 0.1 |  FedAvg <br> FedOpt <br> FedAvg <br> FedOpt | 77.74% <br> 85.17% <br> 76.66% <br> 79.80% | 0.05 | 200 |



Feel free to contact us or open issue!

Corresponding Author: Tiantian Feng, University of Southern California

Email: tiantiaf@usc.edu

### Related Citation

```
@article{feng2023fedmultimodal,
  title={FedMultimodal: A Benchmark For Multimodal Federated Learning},
  author={Feng, Tiantian and Bose, Digbalay and Zhang, Tuo and Hebbar, Rajat and Ramakrishna, Anil and Gupta, Rahul and Zhang, Mi and Avestimehr, Salman and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2306.09486},
  year={2023}
}
```

FedMultimodal also uses the code from our previous work:
```
@inproceedings{zhang2023fedaudio,
  title={Fedaudio: A federated learning benchmark for audio tasks},
  author={Zhang, Tuo and Feng, Tiantian and Alam, Samiul and Lee, Sunwoo and Zhang, Mi and Narayanan, Shrikanth S and Avestimehr, Salman},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```