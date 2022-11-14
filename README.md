# FedMultimodal
#### FedMutimodal is an open source project for researchers exploring multimodal applications in Federated Learning setup

The framework figure:

<div align="center">
 <img src="img/FedMultimodal.jpg" width="750px">
</div>



## Applications supported
* Speech Emotion Recognition
* Multimedia Action Recognition
* Human Activity Recognition

### Speech Emotion Recognition (Natural Split)

Dataset | Modality | Paper | Num. of Train Speakers | Hours of data | Best UAR (Federated) | Learning Rate | Global Epoch
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
MELD | A+T+V | [arxiv](https://arxiv.org/abs/1810.02508) | 86 |     | Bert:55.02% <br> Mobilebert:53.43% | 0.01 | 300
MSP-Podcast | A+T(ASR) | [TAFFC'19](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Lotfian_2019_3.pdf) | >200 |    |


### Multimedia Action Recognition (Manual Split)

Dataset | Modality | Paper | Num. of Clients | Alpha | Best Top-1 Acc (Federated) | Best Top-5 Acc (Federated) | Learning Rate | Global Epoch
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
UCF101 <br> | A+V <br> | [arxiv](https://arxiv.org/abs/1212.0402) <br> | 100 <br> | 1.0 <br> 0.5 <br> 0.1 | 70.20% <br> 69.26% <br> 66.63% | 93.66% <br> 93.56% <br> 92.13% | 0.05 <br> | 300 <br> 
MIT51 (Subset of MIT) | A+V | [arxiv](https://arxiv.org/abs/1801.03150) | 1000 | 1.0 <br> 0.5 <br> 0.1 | 35.12% <br> 35.21% <br> 33.46% | 66.12% <br> 66.16% <br> 63.54% | 0.1 | 500

Dataset | Modality | Paper | Num. of Train Clients | Alpha | Best UAR (Federated) | Learning Rate | Global Epoch
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
UCI-HAR | Acc+Gyro | [UCI-Data](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 86 | 1.0 <br> 0.5 <br> 0.1 | 80.10% <br> 80.27% <br> 79.73% | 0.01 | 300



Feel free to contact us!

Tiantian Feng, University of Southern California

Email: tiantiaf@usc.edu