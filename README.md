# DISCODE
![model](https://github.com/user-attachments/assets/bb7e7706-8a8f-4491-b834-315ab701b5d1)

**DISCODE** (**D**eep learning-based **I**terative pipline to analyze **S**pecificity of **CO**factors and to **D**esign **E**nzyme) is a transformer-based NAD/NADP classification model. The model uses ESM-2 language model for amino acid embedding, finally represents the probability of NAD and NADP. This code can also be run on Google Colaboratory.

## Repository Overview
This structure helps users easily navigate and understand the DISCODE repository.
  - **discode.yaml** : A YAML file that lists all the necessary dependencies for creating a Conda environment.
  - **discode/models.py** : Contains the model architecture and loading functions.
  - **discode/utils.py** : Utility functions such as sequence processing, model inference, visualization, and mutant design tools.
  - **example/example.ipynb** : A Jupyter notebook for identifying and designing cofactor-switching mutants.
  - **weights/weights.pt** : The weight file that needs to be loaded for using the model.

## Installation
### Manual install
**Note : This code was developed in Linux, and has been tested in Ubuntu 18.04 and 20.04 with Python 3.8.<br>**
**While the code requires a conda environment for easier use, it can also be used without conda by installing the dependencies listed in the yaml file via pip.<br>**
1. Clone github repository
```
git clone https://github.com/SBML-Kimlab/DISCODE.git
```
2. Create and activate virtual environment
```
cd DISCODE
conda env create -f discode.yaml
conda activate discode
```
### Running DISCODE on Google Colaboratory workspace
https://colab.research.google.com/drive/1Gm9QrmYHqLfUqZY0xz6jqfIOZSoqkoiP?usp=sharing<br>

## Usage
**Preparation**
```python
from discode import models, utils

model_path = "weights/weights.pt" # please specify the model weight path
model = models.load(model_path) # if gpu available, it will automatically load on gpu
model.eval() # Model must be specified "eval"

# Q9K3J3 is Streptomyces coelicolor malate dehydrogenase
name, sequence = "Q9K3J3", "MTRTPVNVTVTGAAGQIGYALLFRIASGQLLGADVPVKLRLLEITPALKAAEGTAMELDDCAFPLLQGIEITDDPNVAFDGANVALLVGARPRTKGMERGDLLEANGGIFKPQGKAINDHAADDIKVLVVGNPANTNALIAQAAAPDVPAERFTAMTRLDHNRALTQLAKKTGSTVADIKRLTIWGNHSATQYPDIFHATVAGKNAAETVNDEKWLADEFIPTVAKRGAAIIEARGASSAASAANAAIDHVYTWVNGTAEGDWTSMGIPSDGSYGVPEGIISSFPVTTKDGSYEIVQGLDINEFSRARIDASVKELSEEREAVRGLGLI"
```
**Example of classification**
```python
# Predict label of wildtype sequence
# The sequence data will be preprocessed with ESM2-t12 model
data = utils.tokenize_and_dataloader(name, sequence)

# The processed data will be transferred into model, and predict the probability, attention weights, and outlier residues
# The default threshold for selecting outliers is set to 2S
outlier_idx, probability, predicted_label, _name, attention_weights = utils.model_prediction(data, model, threshold="2S")
# The first column of probability is NAD probability, and the second column is NADP probability
print(f"The label probability of NAD is {probability.detach().numpy()[0]:.3f}, NADP is {probability.detach().numpy()[1]:.3f}")
# The label probability of NAD is 0.999, NADP is 0.001
```
**Plot the attention sum and outlier residues**
```python
# The default threshold for selecting outliers is set to 2S
utils.plot_attention_sum(attention_weights, sequence, threshold="2S")
```

## Designing Cofactor-Switching Mutants:
**Ipynb example of a designing pipeline for cofactor switching mutants is in example/example.ipynb.<br>**
**We provide various thresholds for outlier selection to accommodate user's purposes and convenience.<br>**
[Options]
  - max_num_mutation: set the maximum number of mutations to yield cofactor switching mutant (default=3)
  - max_num_solution: set the maximum number of solutions to return (default=50)
  - prob_thres: set a probability threshold for cofactor specificity reversal (default=0.5)
  - pickle_path: directory where a pickle file is saved (default='.')
  - sequence: a protein sequence aimed at changing cofactor specificity
  - name: sequence id (default='unknown')
  - threshold (default="2S") : Specifies the method for selecting outliers
    * Standard deviation-based thresholds : 1S (1 sigma), 2S (2 sigma), 3S (3 sigma)
    * Percentile-based thresholds : IQR (Interquartile Range), P90 (percentile 90), P95 (percentile 95), P99 (percentile 99)
  - mode (default=iter_num):
    * iter_prob : scan all combinations of mutations guided by attention analysis and return those for optimal probabilities (exhaustively calculate most probable designs).
    * iter_num : scan all combinations of residues guided by attention analysis. However, if a cofactor switching design is obtained, scan only to combinations from the same number of mutations (exhaustively calculate a minimal requirement of designs).
    * shortest : scan minimum number of combinations by selecting mutations that show optimal probability changes (fastest search).

**The results generated by each mutation step will be saved on pickle_path as {name_mode_mutation_step}.pkl**
```python
utils.scan_switch_mutation(model = model,
                           max_num_mutation = 3,
                           name = name,
                           sequence = sequence,
                           mode = "iter_num",
                           threshold = "2S",)
```

## Contact
If you have any questions, problems or suggestions, please contact [us](https://sites.google.com/view/systemskimlab/home).

## Citation
Kim J., Woo J., Park J.Y., Kim K.J., Kim D. Deep learning for NAD/NADP cofactor prediction and engineering using transformer attention analysis in enzymes. Metab. Eng. 2025; 87:86-94. https://doi.org/10.1016/j.ymben.2024.11.007.

## Reference
1. A. Vaswani et al., Attention Is All You Need. Adv Neur In 30 (2017).
2. Z. M. Lin et al., Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123-1130 (2023).

