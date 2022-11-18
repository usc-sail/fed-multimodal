# FedMultimodal
#### FedMutimodal is an open source project for researchers exploring multimodal applications in Federated Learning setup

The framework figure:

<div align="center">
 <img src="img/FedMultimodal.jpg" width="750px">
</div>


## Applications supported
* #### Cross-Device Applications
    * Speech Emotion Recognition
    * Multimedia Action Recognition
    * Human Activity Recognition
* #### Cross-silo Applications (Mainly Medical Settings)
    * Sleep Monitoring
    * ECG classification
    * Medical Imaging

## Cross-Device Applications
### Speech Emotion Recognition (Natural Split)

Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Best UAR (Federated) | Learning Rate | Global Epoch
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
MELD | A+T+V | [arxiv](https://arxiv.org/abs/1810.02508) | 4 | 86 | Natural | Bert:55.51% <br> Mobilebert:52.42% | 0.01 | 300
MSP-Podcast | A+T(ASR) | [TAFFC'19](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Lotfian_2019_3.pdf) | 6 | >200 |    |


### Multimedia Action Recognition (Manual Split)

Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | Best Top-1 Acc (Federated) | Best Top-5 Acc (Federated) | Learning Rate | Global Epoch | Fold
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
UCF101 | A+V | [arxiv](https://arxiv.org/abs/1212.0402) | 51 | 200 | Manual | 1.0 <br> 0.25 <br> 0.1 | 69.87% <br> 68.25% <br> 66.42% | 94.57% <br> 93.61% <br> 93.81% | 0.1 <br> | 300 <br> | 3 folds from dataset
MIT10 (Subset of MIT) | A+V | [arxiv](https://arxiv.org/abs/1801.03150) | 10 | 200 | Manual | 1.0 <br> 0.25 <br> 0.1 | 55.90% <br> 50.56% <br> 45.51% | 93.89% <br> 92.87% <br> 85.11% | 0.1 | 300 | 3 folds with 3 seeds
MIT51 (Subset of MIT) | A+V | [arxiv](https://arxiv.org/abs/1801.03150) | 51 | 1000 | Manual | 1.0 <br> 0.25 <br> 0.1 | 34.17% <br> 32.48% <br> 32.17% | 64.76% <br> 63.71% <br> 62.04% | 0.1 | 300 | 3 folds with 3 seeds

### Human Acitivity Recognition (Manual Split/Natural Split)
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | Best UAR (Federated) | Learning Rate | Global Epoch | Fold |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
UCI-HAR | Acc+Gyro | [UCI-Data](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 6 | 210 | Natural+Manual | 1.0 <br> 0.25 <br> 0.1 | 78.60% <br> 78.27% <br> 76.62% | 0.1 | 300 | 5 folds with 5 seeds
Extrasensory | Acc+Gyro | [UCSD-Extrasensory](http://extrasensory.ucsd.edu/) | 6 | 40> | Natural | - | 31.23% | 0.1 | 300 | 5 folds with 5 seeds

## Cross-silo Applications

### ECG Classification (Manual Split/Natural Split)
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | Best Macro-F1 (Federated) | Learning Rate | Global Epoch | Fold |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
PTB-xl | I,II,III,AVL,AVR,AVF+ V1,V2,V3,V4,V5,V6 | [Sci. Data](https://www.nature.com/articles/s41597-020-0495-6) | 5 | 35 (<10 recording Discard) | Natural | - | 62.94% | 0.05 | 200 | 5 folds with 5 seeds

Feel free to contact us!

Tiantian Feng, University of Southern California

Email: tiantiaf@usc.edu