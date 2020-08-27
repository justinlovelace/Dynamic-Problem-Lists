# Dynamic Problem Lists
Code for the paper [Dynamically Extracting Outcome-Specific Problem Lists from Clinical Notes with Guided Multi-Headed Attention](https://arxiv.org/abs/2008.01197). This repo is currently a work in progress as I continue to clean up the code. Feel free to create an issue if there are any problems.

## Data Preparation
Set the appropriate data paths in `resources/config.yml`.

```
python extract_labels.py
python cross_val_splits.py
python word_embeddings.py
```

## Training
Set the training hyperparameters and define the task in `resources/params.json`.
```
python train.py
```

### Acknowledgements
We adapted publicly available code from the following sources:
- Preprocessing and modeling code from the [DeepEHR](https://github.com/NYUMedML/DeepEHR) repo
- Modeling code from the [CAML](https://github.com/jamesmullenbach/caml-mimic) repo
- Boilerplate code (e.g. loading data, training) from [Stanford's CS230 Code Examples](https://github.com/cs230-stanford/cs230-code-examples) repo