##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import os
import torch
from utils.general_utils import log_print, models_dir, text_dir, init_entry_point

# Dataset
from datasets_src.dataset_config import DatasetConfig
from dataset_builders.dataset_builder_creator import create_dataset_builder
from dataset_builders.concreteness_dataset import ConcretenessDatasetBuilder

# Metric
from metrics.concreteness_prediction_metric import ConcretenessPredictionMetric

# Models
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper, TextOnlyCountsModelWrapper
from models_src.wrappers.concreteness_supervised_model_wrapper import ConcretenessSupervisedModelWrapper


def main_concreteness_evaluation(write_to_log, model_type, model_name):
    function_name = 'main_concreteness_evaluation'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    coco_builder, slice_str, multi_label = create_dataset_builder(dataset_name)

    training_set_config = DatasetConfig(1)
    training_set, _, _ = coco_builder.build_dataset(training_set_config)
    token_count = training_set.get_token_count()
    concreteness_dataset = ConcretenessDatasetBuilder(1).build_dataset()
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Testing...')

    text_model_dir = os.path.join(models_dir, text_dir)
    if model_type == 'text_only':
        model_name = 'text_only_baseline'
    if model_name is None:
        log_print(function_name, 0, 'Please enter model name using the flag --model_name')
        return
        
    if model_type == 'multimodal_clustering':
        model = TextCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    elif model_type == 'text_only':
        model_name = 'text_only_baseline'
        model = TextOnlyCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    elif model_type == 'supervised_concreteness':
        model = ConcretenessSupervisedModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    else:
        log_print(function_name, 0, 'Incorrect model type, please use --model_type <MODEL_TYPE>, where <MODEL_TYPE> is one of [multimodal_clustering, text_only, supervised_concreteness]')

    metric = ConcretenessPredictionMetric(model, concreteness_dataset, token_count)
    log_print(function_name, 1, metric.report())

    log_print(function_name, 0, 'Finished testing')
