# GRML_analyses

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/iagea/GRML_analyses/graphs/commit-activity)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10607644.svg)](https://doi.org/10.5281/zenodo.10607644)

This library is designed to execute a suite of 100 random forest classification models leveraging the Mondrian cross-conformal prediction method, as outlined in our published paper.

Additionally, the library offers the functionality to compute and retrieve scores for QED (Quantitative Estimate of Drug-likeness), NIBR (Novartis Institutes for BioMedical Research), and MolSkill. It also facilitates the estimation of predicted pEC50 values.

### Installation

Install [GRML_analyses](https://github.com/Iagea/GRML_analyses)
```
git clone https://github.com/Iagea/GRML_analyses/tree/main
cd GRML_analyses
conda env create -f environment.yml
conda activate GRML_analyses
pip install -e .
```

Install [MolSkill](https://github.com/microsoft/molskill/tree/main)
```
git clone https://github.com/microsoft/molskill.git
cd molskill
pip install .
```

# Examples

```
python predict_GR_class.py ./data/test.csv -c morph_smiles --CL 0.8 --other_scores 1
```
