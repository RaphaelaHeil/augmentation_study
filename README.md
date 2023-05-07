# A Study of Augmentation Methods for Handwritten Stenography Recognition

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/637464009.svg)](https://zenodo.org/badge/latestdoi/637464009)



### Raphaela Heil [:envelope:](mailto:raphaela.heil@it.uu.se), Eva Breznik

Code for the [IbPRIA 2023](http://www.ibpria.org/2023/) paper **"A Study of Augmentation Methods for Handwritten Stenography Recognition"**

Preprint and supplementary material available here: [https://arxiv.org/abs/2303.02761](https://arxiv.org/abs/2303.02761)

## Table of Contents
1. [Requirements](#requirements)
2. [Code](#code)
    1. [Training](#training)
    2. [Testing](#testing)
3. [Citation](#citation)
4. [Acknowledgements](#acknowledgements)

## Requirements

See [requirements.txt](requirements.txt). 

The BatchRenorm package can be installed via: 

```shell
pip install git+https://github.com/ludvb/batchrenorm@master
```

## Code

### Training

```shell
python -m aug.run -file <path-to-config-file> -section <config-section-name>
```

For example: 
```shell
python -m aug.run -file ../config.cfg -section BASELINE
```

### Testing

```shell
python -m aug.run -file <path-to-config-file> -section <config-section-name> -test
```

For testing, the config file path should point to a config file in a result directory, e.g.: 

```shell
Results
├── experiment_001
    ├── config.cfg # <-- here
    ├── info.log
    ├── train.log
    ├── ... 
├── experiment_002
    ├── config.cfg
    ├── info.log
    ├── train.log
    ├── ... 
├── ...
```
## Citation
[IbPRIA 2023](http://www.ibpria.org/2023/)

```
@INPROCEEDINGS{heilbreznik2023aug,
  author={Heil, Raphaela and Breznik, Eva},
  booktitle={11th Iberian Conference on Pattern Recognition and Image Analysis (IbPRIA 2023)},
  title={{A Study of Augmentation Methods for Handwritten Stenography Recognition}},
  year={2023},
  pubstate={to appear}}
```

## Acknowledgements
This work is partially supported by Riksbankens Jubileumsfond (RJ) (Dnr P19-0103:1). The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at Chalmers Centre for Computational Science and Engineering (C3SE) partially funded by the Swedish Research Council through grant agreement no. 2022-06725. Author E.B. is partially funded by the Centre for Interdisciplinary Mathematics, Uppsala University, Sweden.






