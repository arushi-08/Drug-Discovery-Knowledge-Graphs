# Drug Discovery using Knowledge Graphs

## Project Overview

This project focuses on enhancing drug discovery through the use of knowledge graph modeling. By predicting likely missing links in a directed heterogeneous multigraph medication-gene network, we aim to improve the explainability and effectiveness of drug discovery processes.

## Objectives

1. Predict the likely missing links in Hetionet KG medication-gene network.
2. Improve drug discovery and explainability by leveraging advanced knowledge graph embeddings.

## Technologies 

Python
Libraries and Frameworks: PyKeen
Models: TransE, TransH, RotatE

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



