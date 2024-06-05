from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
import torch
import pandas as pd

import os

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first and second GPUs


from pykeen.constants import PYKEEN_CHECKPOINTS
checkpoint = torch.load(PYKEEN_CHECKPOINTS.joinpath(
    '/afs/cs.pitt.edu/usr0/ars539/.data/pykeen/checkpoints/rotate-checkpoint.pt')
                       )

print("Loading data")

# tf = TriplesFactory.from_path(HETIONET_DATA_PATH)
# training, testing, validation = tf.split([.8, .1, .1])
HETIONET_TRAIN_PATH = '/afs/cs.pitt.edu/usr0/ars539/biology_project/hetionet_data_bidir_train_entity.txt'
HETIONET_VAL_PATH = '/afs/cs.pitt.edu/usr0/ars539/biology_project/hetionet_data_bidir_val_entity.txt'
HETIONET_TEST_PATH = '/afs/cs.pitt.edu/usr0/ars539/biology_project/hetionet_data_bidir_test_entity.txt'

training = TriplesFactory.from_path(
    path=HETIONET_TRAIN_PATH,
    entity_to_id=checkpoint['entity_to_id_dict'],
    relation_to_id=checkpoint['relation_to_id_dict'],
)
validation = TriplesFactory.from_path(
    path=HETIONET_VAL_PATH,
    entity_to_id=checkpoint['entity_to_id_dict'],
    relation_to_id=checkpoint['relation_to_id_dict'],
)
testing = TriplesFactory.from_path(
    path=HETIONET_TEST_PATH,
    entity_to_id=checkpoint['entity_to_id_dict'],
    relation_to_id=checkpoint['relation_to_id_dict'],
)

print("Starting pipeline")
result = hpo_pipeline(
    study_name='rotate_hetionet_hpo',
    training=training,
    testing=testing,
    validation=validation,
    pruner="MedianPruner",
    sampler="tpe",
    model='RotatE',
    model_kwargs={
        "random_seed": 42,
    },
    model_kwargs_ranges=dict(
        embedding_dim=dict(type=int, low=100, high=300, q=100),
    ),
    negative_sampler_kwargs_ranges=dict(
        num_negs_per_pos=dict(type=int, low=1, high=100),
    ),
    stopper='early',
    n_trials=30,
    training_loop="sLCWA",
    training_kwargs=dict(
        num_epochs=500,
        drop_last=False,
#         checkpoint_name='rotate-checkpoint-2.pt',
        checkpoint_frequency=10,
     ),
    save_model_directory='.data/pykeen/checkpoints/',
    evaluator_kwargs={"filtered": True, "batch_size":64},
)
result.save_to_directory('./train_rotate_model_20231111')

