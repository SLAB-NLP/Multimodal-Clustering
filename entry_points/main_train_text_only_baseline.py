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
from models_src.model_configs.text_only_model_config import TextOnlyModelConfig

# Executors
from executors.trainers.text_only_model_trainer import TextOnlyModelTrainer


""" The entry point for training a text-only baseline. """


def main_train_text_only_baseline(write_to_log):
    function_name = 'main_train_text_only_baseline'
    timestamp = init_entry_point(write_to_log)

    model_config = TextOnlyModelConfig()
    log_print(function_name, 0, str(model_config))

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)
    training_set_config = DatasetConfig(1)
    training_set, _, _ = dataset_builder.build_dataset(training_set_config)
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Training model...')
    model_root_dir = os.path.join(project_root_dir, timestamp)
    trainer = TextOnlyModelTrainer(model_root_dir, training_set, model_config, 1)
    trainer.train()
    log_print(function_name, 0, 'Finished training model')
