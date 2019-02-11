# LeafNATS - A Learning Framework for Neural Abstractive Text Summarization

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/tshi04/LeafNATS/graphs/contributors)
[![image](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/tshi04/LeafNATS/issues)
[![image](https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat)](https://arxiv.org/abs/1812.02303)

This repository is a pytorch implementation of a learning framework for implementing different models for the neural abstractive text summarization and beyond. 
It is an extension of [NATS](https://github.com/tshi04/NATS) toolkit, which is a toolkit for Neural Abstractive Text Summarization. 
The goal of this framework is to make it convinient to try out new ideas in abstractive text summarization and other language generation tasks.

Live System Demo http://dmkdt3.cs.vt.edu/leafNATS/

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
- ```Check:``` Go to [examples](https://github.com/tshi04/LeafNATS/tree/master/LeafNATS/examples) to check models we have implemented.

- ```Import:``` In run.py, import the example you want to try.

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

