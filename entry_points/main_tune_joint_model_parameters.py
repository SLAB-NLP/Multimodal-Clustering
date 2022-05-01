##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

# General
import os
from utils.general_utils import log_print, init_entry_point, project_root_dir

# Dataset
from dataset_builders.dataset_builder_creator import create_dataset_builder
from datasets_src.dataset_config import DatasetConfig

# Model
from models_src.model_configs.cluster_model_config import ClusterModelConfig

# Executors
from executors.trainers.multimodal_clustering_model_trainer import MultimodalClusteringModelTrainer


""" The entry point for tuning our main model: we remove the validation data from the training set, and evaluate on the
    the validation data.
"""


def main_tune_joint_model_parameters(write_to_log):
    function_name = 'main_tune_joint_model_parameters'
    timestamp = init_entry_point(write_to_log)

    model_config = ClusterModelConfig()
    log_print(function_name, 0, str(model_config))

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)

    training_set_config = DatasetConfig(1, exclude_val_data=True)
    training_set, _, _ = dataset_builder.build_dataset(training_set_config)
    token_count = training_set.get_token_count()

    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Training model...')
    model_root_dir = os.path.join(project_root_dir, timestamp)
    trainer = MultimodalClusteringModelTrainer(model_root_dir, training_set, 2, model_config, token_count, 1)
    trainer.train()
    log_print(function_name, 0, 'Finished training model')
