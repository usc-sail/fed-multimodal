# UCF101 data set

### FedAvg Baseline

Dataset | Modality | Paper | Num. of Clients | Split | Alpha | Best Top-1 Acc (Federated) | Best Top-5 Acc (Federated) | Learning Rate | Global Epoch | Fold
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
UCF101 | A+V | [arxiv](https://arxiv.org/abs/1212.0402) | 200 | Manual | 1.0 <br> 0.25 <br> 0.1 | 69.87% <br> 68.25% <br> 66.42% | 94.57% <br> 93.61% <br> 93.81% | 0.1 <br> | 300 <br> | 3 folds from dataset


### Training with missing modality

Modality Missing Ratio | Alpha | Best Top-1 Acc (Federated) | Best Top-5 Acc (Federated) | Learning Rate | Global Epoch | Fold
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
10% | 1.0 <br> 0.25 <br> 0.1 | 69.80% <br> 66.67% <br> 66.53% | 93.64% <br> 92.94% <br> 93.01% | 0.1 <br> | 300 <br> | 3 folds from dataset
30% | 1.0 <br> 0.25 <br> 0.1 | 66.64% <br> 62.70% <br> 61.56% | 93.11% <br> 91.80% <br> 91.64% | 0.1 <br> | 300 <br> | 3 folds from dataset
50% | 1.0 <br> 0.25 <br> 0.1 | 58.21% <br> 54.57% <br> 49.66% | 89.74% <br> 87.80% <br> 85.23% | 0.1 <br> | 300 <br> | 3 folds from dataset