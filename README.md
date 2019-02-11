# LeafNATS - A Learning Framework for Neural Abstractive Text Summarization

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/tshi04/LeafNATS/graphs/contributors)
[![image](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/tshi04/LeafNATS/issues)
[![image](https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat)](https://arxiv.org/abs/1812.02303)

This repository is a pytorch implementation of a training framework of seq2seq models for the neural abstractive text summarization and beyond. It is an extension of [NATS](https://github.com/tshi04/NATS) which is a toolkit for Neural Abstractive Text Summarization. The goal of this framework is to make it convinient to try out new ideas in abstractive text summarization and other language generation models.



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

In this survey, we run an extensive set of experiments with NATS on the following datasets. Here, we provide the link to CNN/Daily Mail dataset and data processing codes for Newsroom and Bytecup2018 datasets. 
- [CNN/Daily Mail](https://github.com/abisee/pointer-generator)
- [Newsroom](https://github.com/tshi04/LeafNATS/tree/master/LeafNATS/tools/newsroom_process)
- [Bytecup2018](https://github.com/tshi04/LeafNATS/tree/master/LeafNATS/tools/bytecup_process)
The preprocess data will be shared upon request.

In the dataset, \<s\> and \</s\> is used to separate sentences. \<sec\> is used to separate summaries and articles. We did not use the json format because it takes more space and be difficult to transfer between servers.


