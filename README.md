# AlphaDesign: A graph protein design method and benchmark on AlphaFoldDB

<img src=".github/overview.svg" width="800" >

## Dataset
The original dataset can be downloaded from [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/.)

The processed dataset can be downloaded from [google drive](https://drive.google.com/drive/folders/1TeojgosleXo3j4sF41vvOjCbOthPQfKm?usp=sharing).

## Requirement
```
PyTorch>=1.10.2
torch-geometric==2.0.3
```

## Getting Started
```
cd ex1_AlphaDesign
python main.py --method AlphaDesign --data_name UP000000437_7955_DANRE_v2 --ex_name AlphaDesign_DANRE
```

## Citing AlphaDesign

If you use AlphaDesign in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@article{gao2022alphadesign,
  title={AlphaDesign: A graph protein design method and benchmark on AlphaFoldDB},
  author={Gao, Zhangyang and Tan, Cheng and Li, Stan and others},
  journal={arXiv preprint arXiv:2202.01079},
  year={2022}
}
```