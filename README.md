# GRML_analysis

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/iagea/GRML_analyses/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/iagea/GRML_analyses.svg)](https://GitHub.com/iagea/GRML_analyses/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/iagea/GRML_analyses.svg)](https://GitHub.com/iagea/GRML_analyses/graphs/contributors/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10051801.svg)](https://doi.org/10.5281/zenodo.10051801)

This library will perform a set of 100 random forest classification models using Mondrian cross-conformal prediction, as described on our paper [](). 

It also gives the option to obtain QED, NIBR and MolSkill scores. As well as the predicted pEC50.

### Installation

Install [GRML_analyses](https://github.com/Iagea/GRML_analyses)
```
git clone https://github.com/Iagea/GRML_analyses/tree/main
cd GRML_analyses
conda env create -f environment.yml
conda activate GRML_analyses
pip install -e .
cd ../
```

Install [MolSkill](https://github.com/microsoft/molskill/tree/main)
```
git clone https://github.com/microsoft/molskill.git
cd molskill
pip install .
```

# Examples

```
python predict_GR_class.py test.csv -c morph_smiles --CL 0.8 --other_scores 1
```
\