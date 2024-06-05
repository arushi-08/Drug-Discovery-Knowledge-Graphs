import os
import datetime

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch

from config import HETIONET_DATA_PATH

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use the first and second GPUs


# from pykeen.constants import PYKEEN_CHECKPOINTS
# checkpoint = torch.load(PYKEEN_CHECKPOINTS.joinpath(
#     HETIONET_DATA_PATH
# )
                            

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
result.save_to_directory(f"./train_transe_model_{datetime.datetime.today().strftime('%Y%m%d')}")
