from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch

import os

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use the first and second GPUs


# from pykeen.constants import PYKEEN_CHECKPOINTS
# checkpoint = torch.load(PYKEEN_CHECKPOINTS.joinpath(
#     '/afs/cs.pitt.edu/usr0/ars539/.data/pykeen/checkpoints/best-model-weights-c909159d-4f5e-4d6e-95aa-7d5e6cf6b043.pt'
# )
                            
HETIONET_DATA_PATH = (
    '/afs/cs.pitt.edu/usr0/ars539/biology_project/hetionet_data.txt'
)

tf = TriplesFactory.from_path(HETIONET_DATA_PATH)
training, testing, validation = tf.split([.8, .1, .1])
result = pipeline(
    random_seed=1247395728,
    training=training,
    testing=testing,
    validation=validation,
    model='TransE',
    stopper='early',
    dimensions=304,
    stopper_kwargs = {
        "frequency": 5,
        "patience": 20,
        "relative_delta": 0.002,
        "metric": "hits@10",
    },
    training_loop='slcwa',
    optimizer=torch.optim.Adam,
    lr_scheduler='ExponentialLR',
    lr_scheduler_kwargs={
        "gamma": 0.1,
    },
    negative_sampler_kwargs={
        "num_negs_per_pos":61
    },
    loss='NSSA',
    training_kwargs=dict(
        num_epochs=500,
        checkpoint_name='transe-checkpoint.pt',
        checkpoint_frequency=10,
     ),
)
result.save_to_directory('./train_transe_model_20230731')