

# Multi-label Learning for Appliances Recognition in NILM using Fryze-Current Decomposition and Convolutional Neural Network.

This repository is the official implementation of [Multi-label Learning for Appliances Recognition in NILM using Fryze-Current Decomposition and Convolutional Neural Network](). 

<img src="block.png" width="80%" height="50%">

The paper present a multi-label learning strategy for appliance recognition in NILM. The proposed approach associates multiple appliances to an observed aggregate current signal. We first demonstrate that for aggregated measurements, the use of activation current as an input feature offers improved performance compared to the regularly used V-I binary image feature. Second, we apply the Fryze power theory and Euclidean distance matrix as pre-processing steps for the multi-label classifier.
## Requirements

- python
- numpy
- scipy
- pandas
- matplotlib
- tqdm
- torch
- sklearn
- seaborn
- [iterstrat]
- [skmultilearn]()
- joblib
- [palettable]
  


## Training

To train the model(s) in the paper, run this command in src directory:

```train
python multilabel.py
```


## Evaluation

The script used to analyse results and produce visualisation presented in this paper can be found in notebook directory
 
 - ResultsAnalysis notebook provides scripts for results and error analysis.
 - VisualizePaper and VisualizeFeature notebook provide scripts for reproducing most of the figure used in this paper.


## Results

Our model achieves the following performance on PLAID 2020 version dataset :



| Model name         | V-I binary image  | Proposed approach |
| ------------------ |---------------- | -------------- |
| MLkNN  model |     0.779±0.028        |      0.833±0.022      |
| Proposed CNN model  |     0.826±0.024         |      0.94±0.015       |



## Contributing

