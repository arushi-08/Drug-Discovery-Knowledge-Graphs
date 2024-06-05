# Drug Discovery using Knowledge Graphs

## Project Overview

This project focuses on enhancing drug discovery through the use of knowledge graph modeling. By predicting likely missing links in a directed heterogeneous multigraph medication-gene network, we aim to improve the explainability and effectiveness of drug discovery processes.

## Objectives

1. Predict the likely missing links in [Hetionet KG](https://github.com/hetio/hetionet?tab=readme-ov-file) biology network.
2. Improve drug discovery and explainability by leveraging advanced knowledge graph embeddings.

## Technologies 

Python
Libraries and Frameworks: [PyKeen](https://pykeen.readthedocs.io/en/stable/) \
Models: [TransE](https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html), [TransH](https://ojs.aaai.org/index.php/AAAI/article/view/8870), [RotatE](https://deepai.org/publication/rotate-knowledge-graph-embedding-by-relational-rotation-in-complex-space)

### Environment setup

```
git clone https://github.com/arushi-08/Drug-Discovery-Knowledge-Graphs.git
cd Drug-Discovery-Knowledge-Graphs
```

Install depedencies
```
pip install -r requirements.txt
```

Run the TransE KGE Experiments (note: TransH and RotatE experiments are triggered in similar fashion, however these models are bigger):
```
python train_transe_hetionet.py
```



