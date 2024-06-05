import os
import json
import datetime

import pandas as pd
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import RotatE
# from pykeen.models import save_model

from config import HETIONET_DATA_PATH

# Load the pipelineconfig.json file
with open('./train_rotate_model_20231118/best_pipeline/pipeline_config.json') as f:
    config = json.load(f)

# Create a fresh knowledge graph instance
# kg = KnowledgeGraph()

print("Loading data")

tf = TriplesFactory.from_path(HETIONET_DATA_PATH)
training, testing, validation = tf.split([.8, .1, .1])

pd.DataFrame(training.mapped_triples.numpy()).to_csv(
    os.path.join(HETIONET_DATA_PATH, 'hetionet_data_bidir_train.txt'), index=False, sep='\t'
)
pd.DataFrame(validation.mapped_triples.numpy()).to_csv(
    os.path.join(HETIONET_DATA_PATH, 'hetionet_data_bidir_val.txt'), index=False, sep='\t'
)
pd.DataFrame(testing.mapped_triples.numpy()).to_csv(
    os.path.join(HETIONET_DATA_PATH, 'hetionet_data_bidir_test.txt'), index=False, sep='\t'
)
print("Starting pipeline")

result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='RotatE',
    model_kwargs=config['pipeline']['model_kwargs'],
    training_loop='slcwa',
    optimizer=torch.optim.Adagrad,
    lr_scheduler='ExponentialLR',
    lr_scheduler_kwargs={
        "gamma": 0.1,
    },
    stopper='early',
    negative_sampler='basic',
    negative_sampler_kwargs=config['pipeline']['negative_sampler_kwargs'],
    loss=config['pipeline']['loss'],
    loss_kwargs=config['pipeline']['loss_kwargs'],
#     optimizer_kwargs=config['pipeline']['optimizer_kwargs'],
    evaluator_kwargs={
      "batch_size": 1024,
      "filtered": True
    },
    training_kwargs={
      "batch_size": 1024,
      "checkpoint_directory": os.path.join(HETIONET_DATA_PATH,"checkpoints"),
      "checkpoint_frequency": 20,
      "checkpoint_on_failure": True,
      "num_epochs": 500
    },
)
result.save_to_directory("./train_rotate_model_{datetime.datetime.today().strftime('%Y%m%d')}")

print("Completed Training")
