# DISCODE
![discode_img_2@4x](https://github.com/SBML-Kimlab/DISCODE/assets/153895812/b9a46ca6-7727-40a3-b345-61d19af44a37)



**DISCODE** (**D**eep learning-based **I**terative pipline to analyze **S**pecificity of **CO**factors and to **D**esign **E**nzyme) is a transformer-based NAD/NADP classification model. The model uses ESM-2 language model for amino acid embedding, finally represents the probability of NAD and NADP. This code can also be run on Google Colaboratory.

## Repository Overview
This structure helps users easily navigate and understand the DISCODE repository.
  - **discode.yaml** : A YAML file that lists all the necessary dependencies for creating a Conda environment.
  - **discode/models.py** : Contains the model architecture and loading functions.
  - **discode/utils.py** : Utility functions such as sequence processing, model inference, visualization, and mutant design tools.
  - **example/example.ipynb** : A Jupyter notebook for identifying and designing cofactor-switching mutants.
  - **weights/weights.pt** : The weight file that needs to be loaded for using the model.

## Installation
**Note : This code was developed in Linux, and has been tested in Ubuntu 18.04 and 20.04 with Python 3.8.**
**While the code requires a conda environment for easier use, it can also be used without conda by installing the dependencies listed in the yaml file via pip.**
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
data = new_utils.tokenize_and_dataloader(name, sequence)

# The processed data will be transferred into model, and predict the probability, attention weights, and outlier residues
# The default threshold for selecting outliers is set to Z=2
outlier_idx, probability, predicted_label, _name, attention_weights = new_utils.model_prediction(data, model, threshold="Z=2")
# The first column of probability is NAD probability, and the second column is NADP probability
print(f"The label probability of NAD is {probability.detach().numpy()[0]:.3f}, NADP is {probability.detach().numpy()[1]:.3f}")
# The label probability of NAD is 0.999, NADP is 0.001
```
**Plot the attention sum and outlier residues**
```python
# The default threshold for selecting outliers is set to Z=2
new_utils.plot_attention_sum(attention_weights, sequence, threshold="Z=2")
```
**Changes in outlier residues based on supporting different thrersholds**
```python
# The supported thresholds are as follows, with the default set to Z=2.
threshold_list = ["Z=1", "Z=2", "Z=3", "IQR", "0.90", "0.95", "0.99"]
for threshold in threshold_list:
    outlier_idx, _, _, _, _ = new_utils.model_prediction(data, model, threshold=threshold)
    print(f"{threshold} : {len(outlier_idx)} outliers, {outlier_idx}")
# Z=1 : 18 outliers, [ 10  11  13  42  43  44  50  53  86  87  88  89 101 129 156 158 185 242]
# Z=2 : 5 outliers, [13 42 43 44 88]
# Z=3 : 3 outliers, [13 42 43]
# IQR : 31 outliers, [  7   9  10  11  13  16  41  42  43  44  50  53  54  86  87  88  89  98 101 129 156 158 159 168 185 189 236 237 239 242 286]
# 0.90 : 33 outliers, [  7   9  10  11  13  16  41  42  43  44  50  53  54  86  87  88  89  98 101 108 129 156 158 159 168 185 188 189 236 237 239 242 286]
# 0.95 : 17 outliers, [ 10  11  13  42  43  44  53  86  87  88  89 101 129 156 158 185 242]
# 0.99 : 4 outliers, [13 42 43 44]
```

## Designing Cofactor-Switching Mutants:
**Ipynb example of a designing pipeline for cofactor switching mutants is in example/example.ipynb.**

[Options]
  - max_num_mutation: set the maximum number of mutations to yield cofactor switching mutant (default=3)
  - max_num_solution: set the maximum number of solutions to return (default=50)
  - prob_thres: set a probability threshold for cofactor specificity reversal (default=0.5)
  - pickle_path: directory where a pickle file is saved (default='.')
  - sequence: a protein sequence aimed at changing cofactor specificity
  - name: sequence id (default='unknown')
  - mode (default=iterative_num):
    * iterative_prob : scan all combinations of mutations guided by attention analysis and return those for optimal probabilities (exhaustively calculate most probable designs).
    * iterative_num : scan all combinations of residues guided by attention analysis. However, if a cofactor switching design is obtained, scan only to combinations from the same number of mutations (exhaustively calculate a minimal requirement of designs).
    * shortest : scan minimum number of combinations by selecting mutations that show optimal probability changes (fastest search).

**The results generated by each mutation step will be saved on pickle_path as {name_mode_mutation_step}.pkl**
```python
utils.scan_switch_mutation(model = model,
                           max_num_mutation = 3,
                           name = name,
                           sequence = sequence,
                           mode = "iter_num",
                           threshold = "Z=2",)
```

## Contact
If you have any questions, problems or suggestions, please contact [us](https://sites.google.com/view/systemskimlab/home).

## Citation

## Reference
1. A. Vaswani et al., Attention Is All You Need. Adv Neur In 30 (2017).
2. Z. M. Lin et al., Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123-1130 (2023).

