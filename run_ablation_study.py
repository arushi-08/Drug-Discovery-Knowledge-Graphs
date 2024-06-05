import torch
from pykeen.ablation import ablation_pipeline

metadata = dict(title="Ablation Study Over Hetionet for ComplEx.")
models = ["ComplEx"]
losses = ["NSSA"]
training_loops = ["sLCWA"]
optimizers = ["adam"]
create_inverse_triples= [True, False]
stopper = "early"
stopper_kwargs = {
    "frequency": 5,
    "patience": 20,
    "relative_delta": 0.002,
    "metric": "hits@10",
}

model_to_model_kwargs_ranges = {
    "TransE": {
        "embedding_dim": {
            "type": "int",
            "low": 2,
            "high": 4,
            "scale": "power_two"
        }
    },
 }

model_to_training_loop_to_training_kwargs = {
    "TransE": {
        "lcwa": {
            "num_epochs": 300
        }
    },
 }

# model_to_training_loop_to_training_kwargs_ranges= {
#     "TransE": {
#         "lcwa": {
#             "label_smoothing": {
#                 "type": "float",
#                 "low": 0.001,
#                "high": 1.0,
#                 "scale": "log"
#             },
#             "batch_size": {
#                 "type": "int",
#                 "low": 1,
#                 "high": 5,
#                 "scale": "power_two"
#             }
#         }
#     },
# }


model_to_optimizer_to_optimizer_kwargs_ranges= {
    "TransE": {
        "adam": {
            "lr": {
                "type": "float",
                "low": 0.001,
                "high": 0.1,
                "scale": "log"
            }
        },
    },
}

# from pykeen.triples import TriplesFactory
HETIONET_TRAIN_PATH = (
    '/afs/cs.pitt.edu/usr0/ars539/biology_project/hetionet_data_train.txt'
)
HETIONET_VAL_PATH = (
    '/afs/cs.pitt.edu/usr0/ars539/biology_project/hetionet_data_val.txt'
)
HETIONET_TEST_PATH = (
    '/afs/cs.pitt.edu/usr0/ars539/biology_project/hetionet_data_test.txt'
)

datasets = [
    {
        "training": HETIONET_TRAIN_PATH,
        "validation": HETIONET_VAL_PATH,
        "testing": HETIONET_TEST_PATH
    }
]
# Run ablation experiment
ablation_pipeline(
    models=models,
    datasets=datasets,
    losses=losses,
    training_loops=training_loops,
    optimizers=optimizers,
    model_to_model_kwargs_ranges=model_to_model_kwargs_ranges,
    model_to_training_loop_to_training_kwargs=model_to_training_loop_to_training_kwargs,
    model_to_optimizer_to_optimizer_kwargs_ranges=model_to_optimizer_to_optimizer_kwargs_ranges,
    directory="./ablation/",
    best_replicates=5,
    n_trials=2,
    timeout=300,
    metric="hits@10",
    direction="maximize",
    sampler="random",
    pruner="nop",
)