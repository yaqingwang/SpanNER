# Learning from Language Description: Low-shot Named Entity Recognition via Decomposed Framework

This is the implementation of the paper [Learning from Language Description: Low-shot Named Entity Recognition via Decomposed Framework](https://arxiv.org/pdf/2109.05357.pdf). 


## Overview
<img src="./figs/metast.png" width="650"/>

In this work we present SpanNER, which learns from natural language supervision to build Few-shot NER learner. At the same time, such a framework enables the
identification of never-seen entity classes without using in-domain labeled data 
You can find more details of this work in our [paper](https://arxiv.org/pdf/2109.05357.pdf).


## Setup Environment
### Install via pip:

1. create a conda environment running Python 3.7: 
```
conda create -n SpanNER python=3.7
conda activate SpanNER
```

2.  install the required dependencies:
```
pip install -r requirements.txt
```


## Quick start
### Run SpanNER

Training on CoNLL03 </br>
   ```> bash ./scripts/run.sh ```




### Notes and Acknowledgments
The implementation is based on https://github.com/huggingface/transformers <br>
We also used some code from: https://github.com/facebookresearch/BLINK


### How do I cite SpanNER?

```
@inproceedings{wang2021learning,
  title={Learning from Language Description: Low-shot Named Entity Recognition via Decomposed Framework},
  author={Wang, Yaqing and Chu, Haoda and Zhang, Chao and Gao, Jing},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages={1618--1630},
  year={2021}
}
```

