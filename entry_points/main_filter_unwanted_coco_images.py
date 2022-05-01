##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

# General
from utils.general_utils import log_print, init_entry_point

# Dataset
from dataset_builders.dataset_builder_creator import create_dataset_builder


""" The entry point for filtering unwanted images from the MSCOCO dataset. """


def main_filter_unwanted_coco_images(write_to_log):
    function_name = 'main_filter_unwanted_coco_images'
    init_entry_point(write_to_log)

    log_print(function_name, 0, 'Generating dataset_files...')
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)
    log_print(function_name, 0, 'Datasets generated')

    log_print(function_name, 0, 'Filtering from train split...')
    dataset_builder.filter_unwanted_images('train')
    log_print(function_name, 0, 'Finished filtering train split')

    log_print(function_name, 0, 'Filtering from test split...')
    dataset_builder.filter_unwanted_images('test')
    log_print(function_name, 0, 'Finished filtering test split')
