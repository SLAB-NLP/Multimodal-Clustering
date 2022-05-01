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
from dataset_builders.swow_dataset import SWOWDatasetBuilder
from dataset_builders.build_word_list_for_categorization import build_word_list_for_categorization

# Metric
from metrics.word_association_metric import WordAssociationMetric

# Models
from models_src.wrappers.text_model_wrapper import \
    TextCountsModelWrapper, \
    TextOnlyCountsModelWrapper, \
    TextRandomModelWrapper
from models_src.wrappers.word_embedding_clustering_model_wrapper import \
    W2VClusteringWrapper, \
    BERTClusteringWrapper, \
    CLIPClusteringWrapper
from models_src.model_configs.cluster_model_config import ClusterModelConfig


""" This entry point evaluated association strength between pairs of words in a clustering solution. """


def main_association_evaluation(write_to_log, model_type, model_name):
    function_name = 'main_association_evaluation'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    all_words = build_word_list_for_categorization(1)
    swow_dataset = SWOWDatasetBuilder(all_words, 1).build_dataset()
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Testing...')

    text_model_dir = os.path.join(models_dir, text_dir)
    if model_type == 'multimodal_clustering':
        model = TextCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    elif model_type == 'text_only':
        model_name = 'text_only_baseline'
        model = TextOnlyCountsModelWrapper(torch.device('cpu'), None, text_model_dir, model_name, 1)
    elif model_type == 'random':
        model = TextRandomModelWrapper(torch.device('cpu'), ClusterModelConfig(cluster_num=41), 1)
    elif model_type == 'w2v':
        model = W2VClusteringWrapper(torch.device('cpu'), all_words)
    elif model_type == 'bert':
        model = BERTClusteringWrapper(torch.device('cpu'), all_words)
    elif model_type == 'clip':
        model = CLIPClusteringWrapper(torch.device('cpu'), all_words)
    else:
        log_print(function_name, 0, 'Incorrect model type, please use --model_type <MODEL_TYPE>, where <MODEL_TYPE> is one of [multimodal_clustering, text_only, random, w2v, bert, clip]')

    metric = WordAssociationMetric(model, all_words, swow_dataset)
    log_print(function_name, 1, metric.report())

    log_print(function_name, 0, 'Finished testing')
