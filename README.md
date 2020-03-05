# LeafNATS - A Learning Framework for Neural Abstractive Text Summarization

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/tshi04/LeafNATS/graphs/contributors)
[![image](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/tshi04/LeafNATS/issues)
[![image](https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat)](https://arxiv.org/abs/1812.02303)

This playground is a pytorch implementation of a learning framework for implementing different models for the neural abstractive text summarization and beyond. 
It is an extension of [NATS](https://github.com/tshi04/NATS) toolkit, which is a toolkit for Neural Abstractive Text Summarization. 
The goal of this framework is to make it convinient to try out new ideas in abstractive text summarization and other language generation tasks.

### Live System Demo http://dmkdt3.cs.vt.edu/leafNATS/

### Demo Video https://www.youtube.com/watch?v=exLbfFxVFfM&t=25s

### Paper https://www.aclweb.org/anthology/N19-4012

## Requirements

- glob
- argparse
- shutil
- spacy
- pytorch 1.0

## Usuage

#### Scripts

- [Set up GPU, cuda and pytorch](https://github.com/tshi04/LeafNATS/tree/master/LeafNATS/tools/config_server)
- [Install pyrouge and ROUGE-1.5.5](https://github.com/tshi04/LeafNATS/tree/master/LeafNATS/tools/rouge_package)

#### Dataset

We tested different models in LeafNATS on the following datasets. Here, we provide the link to CNN/Daily Mail dataset and data processing codes for Newsroom and Bytecup2018 datasets. The preprocessed data will be available upon request.
- [CNN/Daily Mail](https://github.com/abisee/pointer-generator)
- [Newsroom](https://github.com/tshi04/LeafNATS/tree/master/LeafNATS/tools/newsroom_process)
- [Bytecup2018](https://github.com/tshi04/LeafNATS/tree/master/LeafNATS/tools/bytecup_process)

In the dataset, \<s\> and \</s\> is used to separate sentences. \<sec\> is used to separate summaries and articles. We did not use the json format because it takes more space and be difficult to transfer between servers.

#### Examples

LeafNATS is current under development. A simple way to run models that have already implemented is
- ```Check:``` Check models we have implemented in this directory.

- ```Import:``` In run.py, import the example you want to try. For example ```from nats.pointer_generator_network.main import *```

- ```Training:``` python run.py

- ```Validate:``` python run.py --task validate

- ```Test:``` python run.py --task beam

- ```Rouge:``` python run.py --task rouge

#### Features

- ```Engine``` Training frameworks
- ```Playground``` Models, pipelines, loss functions, and data redirection
- ```Modules``` Building blocks, beam search, word-copy for decoding
- ```Data``` Data pre-process and batcher.

## Pretrained Models and Results
Here is the pretrained model for our live system 
https://drive.google.com/open?id=1A7ODPpermwIHeRrnqvalT5zpr4BCTBi9

## Indices of different models

<p align="left">
  <img src="figure/modelIndex.png" width="600" title="The Model" alt="Cannot Access">
</p>

## Experimental Results
Experimental Results can be found in paper
- [Neural Abstractive Text Summarization with Sequence-to-Sequence Models](https://arxiv.org/pdf/1812.02303.pdf)
- [LeafNATS: An Open-Source Toolkit and Live Demo System for Neural Abstractive Text Summarization](https://www.aclweb.org/anthology/N19-4012)

## Citation

```
@article{shi2018neural,
  title={Neural Abstractive Text Summarization with Sequence-to-Sequence Models},
  author={Shi, Tian and Keneshloo, Yaser and Ramakrishnan, Naren and Reddy, Chandan K},
  journal={arXiv preprint arXiv:1812.02303},
  year={2018}
}
```
```
@inproceedings{shi2019leafnats,
  title={LeafNATS: An Open-Source Toolkit and Live Demo System for Neural Abstractive Text Summarization},
  author={Shi, Tian and Wang, Ping and Reddy, Chandan K},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)},
  pages={66--71},
  year={2019}
}
```
