# DISCODE
![DISCODE](/DISCODE.png)
**DISCODE** (**D**eep learning-based **I**terative pipline to analyze **S**pecificity of **CO**factors and to **D**esign **E**nzyme) is a transformer-based NAD/NADP classification model. The model uses ESM-2 language model for amino acid embedding, finally represents the probability of NAD and NADP. This code can also be run on Google Colaboratory.

## Installation
**Note : This code was developed in Linux, and has been tested in Ubuntu 18.04 and 20.04 with Python 3.8.**
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
**Example of classification**
```python
from discode import models, utils

model_path = "weights/weights.pt" # please specify the model weight path
model = models.load(model_path) # if gpu available, it will automatically load on gpu

name, sequence = "3M6I", "MASSASKTNIGVFTNPQHDLWISEASPSLESVQKGEELKEGEVTVAVRSTGICGSDVHFWKHGCIGPMIVECDHVLGHESAGEVIAVHPSVKSIKVGDRVAIEPQVICNACEPCLTGRYNGCERVDFLSTPPVPGLLRRYVNHPAVWCHKIGNMSYENGAMLEPLSVALAGLQRAGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITDIDEGRLKFAKEICPEVVTHKVERLSAEESAKKIVESFGGIEPAVALECTGVESSIAAAIWAVKFGGKVFVIGVGKNEIQIPFMRASVREVDLQFQYRYCNTWPRAIRLVENGLVDLTRLVTHRFPLEDALKAFETASDPKTGAIKVQIQSLE"
dataloader = utils.tokenize_and_dataloader(name, sequence) # preprocess

outlier_idx, probability, predicted_label, _name, attention_weights = utils.model_processing(dataloader, model)
# The outlier_idx is zero-index
# probability, predicted_label, attention_weights, is output of model, the _name is same with previously declared variable name.

utils.make_max_attention_map(attention_weights)
# This will plot maximum attention map of overall model, in shape of [8,20]

utils.make_attention_sum(attention_weights, outlier_idx, sequence)
# This will plot attention sum, in shape of sequence length L
```

**Jupyter example of a designing pipeline for cofactor switching mutants is in example/example.ipynb.**
[Options]
  - max_num_mutation: the maximum number of mutations to yield cofactor switching mutant (default=3)
  - max_num_solution: the maximum number of solutions to return (default=50)
  - prob_thres: probability threshold for cofactor specificity reversal (default=0.5)
  - pickle_path: directory where a pickle file is saved (default='.')
  - sequence: a protein sequence aimed at changing cofactor specificity
  - name: sequence id (default='unknown', the mutation will be concatenated by "_", so do not use underscore)
  - mode (default=iterative_num):
    * iterative_prob : scan all combinations of mutations guided by attention analysis and return those for optimal probabilities (exhaustively calculate most probable designs).
    * iterative_num : scan all combinations of residues guided by attention analysis. However, if a cofactor switching design is obtained, scan only to combinations from the same number of mutations (exhaustively calculate a minimal requirement of designs).
    * shortest : scan minimum number of combinations by selecting mutations that show optimal probability changes (fastest search).

**The results generated by each mutation step will be saved on pickle_path as {name_mode_mutation_step}.pkl**
```python
utils.scan_switch_mutation(model = model,
                           max_num_mutation = 2,
                           max_num_solution = 20,
                           prob_thres = 0.5,
                           name = name,
                           sequence = sequence,
                           pickle_path = "",
                           mode = "shortest",)
```

## Contact
If you have any questions, problems or suggestions, please contact [us](https://sites.google.com/view/systemskimlab/home).

## Citation
J Kim, J Woo, JY Park, KJ Kim, D Kim. Transformer-based NAD/NADP cofactor analysis pipeline provides insights on classification and engineering of enzymes. (paper under a submission)
