##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import gensim.downloader as api
from dataset_builders.category_dataset import CategoryDatasetBuilder
from dataset_builders.dataset_builder_creator import create_dataset_builder
from datasets_src.dataset_config import DatasetConfig

""" The purpose of this file is to build the list of words on which we'll evaluate categorization metrics.
We want it to be all the words in the MSCOCO dataset, that occur in Fountain's dataset and also in the word2vec
dictionary.
"""


def build_word_list_for_categorization(indent):
    category_dataset = CategoryDatasetBuilder(indent).build_dataset()
    # Filter the category dataset to contain only words with which all of the evaluated models are familiar
    all_words = [x for outer in category_dataset.values() for x in outer]

    # 1. Only words that occurred in the COCO training set
    dataset_name = 'COCO'
    dataset_builder, _, _ = create_dataset_builder(dataset_name)
    training_set_config = DatasetConfig(indent)
    training_set, _, _ = dataset_builder.build_dataset(training_set_config)
    token_count = training_set.get_token_count()
    all_words = [x for x in all_words if x in token_count]

    # 2. Only words with which word2vec is familiar
    w2v_model = api.load("word2vec-google-news-300")
    all_words = [x for x in all_words if x in w2v_model.key_to_index]

    return all_words
