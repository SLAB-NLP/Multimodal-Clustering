##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from utils.general_utils import generate_dataset
from dataset_builders.dataset_builder import DatasetBuilder
import os
from csv import reader


class SWOWDatasetBuilder(DatasetBuilder):
    """ This class build the Small world of words dataset, described in the paper "The “small765world of words” english
        word association norms for766over 12,000 cue words" by De Denye et al.
    """
    
    def __init__(self, word_list, indent):
        super(SWOWDatasetBuilder, self).__init__(indent)
    
        self.output_filename = os.path.join(self.cached_dataset_files_dir, 'swow_dataset')
        self.input_filename = os.path.join(self.datasets_dir, 'swow.en.csv')
        # Since the original dataset is huge, we only build our dataset for a provided list of words
        self.word_dict = {x: True for x in word_list}

    def build_dataset(self, config=None):
        return generate_dataset(self.output_filename, self.generate_swow_dataset_internal)

    def generate_swow_dataset_internal(self):
        self.log_print('Generating small-world-of-words dataset...')

        strength_dict = {}

        with open(self.input_filename, newline='', encoding='utf8') as csvfile:
            swow_reader = reader(csvfile, delimiter='\t')
            for row in swow_reader:
                ''' Words might appear in different orders. For example, 'dog' could be the cue and the participant
                responded using 'cat', but the other way around is also possible. We're not interested in the order so
                we we keep a single entry for each pair (the first word would be the first in alphabetical order).
                When there are multiple entries will sum the strength of all entries. '''
                word1 = sorted([row[1], row[2]])[0]
                word2 = sorted([row[1], row[2]])[1]
                strength = row[3]
                if word1 in self.word_dict and word2 in self.word_dict:
                    if word1 not in strength_dict:
                        strength_dict[word1] = {}
                    if word2 not in strength_dict[word1]:
                        strength_dict[word1][word2] = 0
                    strength_dict[word1][word2] += int(strength)

        return strength_dict
