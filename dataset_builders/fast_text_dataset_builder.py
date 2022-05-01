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
import io
import os
import numpy as np


class FastTextDatasetBuilder(DatasetBuilder):
    """ This class builds the fastText word embedding dataset. """

    def __init__(self, word_list, indent):
        super(FastTextDatasetBuilder, self).__init__(indent)

        self.output_filename = os.path.join(self.cached_dataset_files_dir, 'fast_text')
        self.input_filename = os.path.join(self.datasets_dir, 'wiki-news-300d-1M.vec')

        self.word_list = word_list

    def build_dataset(self, config=None):
        return generate_dataset(self.output_filename, self.generate_fast_text_internal)

    def generate_fast_text_internal(self):
        self.log_print('Generating fast text vectors dataset...')

        if not os.path.isfile(self.input_filename):
            self.log_print('Couldn\'t find file ' + self.input_filename + '. Please download from url https://fasttext.cc/docs/en/english-vectors.html')
            assert False
        fin = io.open(self.input_filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        map(int, fin.readline().split())
        data = {}
        counter = 0
        for line in fin:
            if counter % 10000 == 0:
                self.log_print('Starting token ' + str(counter))
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in self.word_list:
                data[word] = np.array([float(x) for x in tokens[1:]])
            counter += 1
        return data
