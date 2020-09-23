

# Multi-label Learning for Appliances Recognition in NILM using Fryze-Current Decomposition and Convolutional Neural Network.

This repository is the official implementation of [Multi-label Learning for Appliances Recognition in NILM using Fryze-Current Decomposition and Convolutional Neural Network](https://www.mdpi.com/1996-1073/13/16/4154). 

<img src="block.png" width="100%" height="100%">

The paper present a multi-label learning strategy for appliance recognition in NILM. The proposed approach associates multiple appliances to an observed aggregate current signal. We first demonstrate that for aggregated measurements, the use of activation current as an input feature offers improved performance compared to the regularly used V-I binary image feature. Second, we apply the Fryze power theory and Euclidean distance matrix as pre-processing steps for the multi-label classifier.
## Requirements

- python
- numpy
- scipy
- pandas
- matplotlib
- tqdm
- [torch](https://pytorch.org/)
- sklearn
- seaborn
- [iterstrat](https://github.com/trent-b/iterative-stratification): pip install iterative-stratification
- [skmultilearn](http://scikit.ml/): pip install scikit-multilearn
- [joblib](https://joblib.readthedocs.io/en/latest/):
- [palettable](https://jiffyclub.github.io/palettable/): pip install palettable
  


## Training

To train the model(s) in the paper, run this command in src directory:

```train
python run_experiment.py
```


## Evaluation

The script used to analyse results and produce visualisation presented in this paper can be found in notebook directory
 
 - ResultsAnalysis notebook provides scripts for results and error analysis.
 - Visual-paper  notebook provide scripts for reproducing most of the figure used in this paper.


## Results

Our model achieves the following performance on PLAID aggregated dataset :

<img src="result_2.png" width="50%" height="50%">




## Prediction sample
<p float="left">
 <img src="pred_decompose.png" width="250" /> 
  <img src="pred_vi.png" width="250" />
  <img src="pred_decompose_distance.png" width="250" />
</p>

If you find this tool useful and use it (or parts of it), we ask you to cite the following work in your publications:
> Faustine, A.; Pereira, L. Multi-Label Learning for Appliance Recognition in NILM Using Fryze-Current Decomposition and Convolutional Neural Network. Energies 2020, 13, 4154.

``tex
@article{Faustine2020,
  doi = {10.3390/en13164154},
  url = {https://doi.org/10.3390/en13164154},
  year = {2020},
  month = aug,
  publisher = {{MDPI} {AG}},
  volume = {13},
  number = {16},
  pages = {4154},
  author = {Anthony Faustine and Lucas Pereira},
  title = {Multi-Label Learning for Appliance Recognition in {NILM} Using Fryze-Current Decomposition and Convolutional Neural Network},
  journal = {Energies}
}
``

