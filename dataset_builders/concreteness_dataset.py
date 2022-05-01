##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from utils.general_utils import generate_dataset, for_loop_with_reports
from dataset_builders.dataset_builder import DatasetBuilder
from loggable_object import LoggableObject
import os


class ConcretenessDatasetBuilder(DatasetBuilder):
    """ This class builds the concreteness dataset.
        The category dataset maps words to a concreteness value on a scale of 1 to 5, annotated by humans.
    """

    def __init__(self, indent):
        super(ConcretenessDatasetBuilder, self).__init__(indent)
        
        self.output_filename = os.path.join(self.cached_dataset_files_dir, 'concreteness_dataset')
        self.input_filename = os.path.join(self.datasets_dir, 'Concreteness_ratings_Brysbaert_et_al_BRM.txt')

    def build_dataset(self, config=None):
        return generate_dataset(self.output_filename,
                                self.generate_concreteness_dataset_internal,
                                self.input_filename)

    def generate_concreteness_dataset_internal(self, dataset_input_filename):
        self.log_print('Generating concreteness dataset...')

        collector = ConcretenessCollector(self.indent + 1)
        concreteness_fp = open(dataset_input_filename, 'r')
        checkpoint_len = 10000
        for_loop_with_reports(concreteness_fp, None, checkpoint_len,
                              collector.collect_line, collector.file_progress_report)

        return collector.concreteness_dataset


class ConcretenessCollector(LoggableObject):
    """ This class parses the file containing the concreteness values and creates the dataset object.
        The assumed file is the dataset by Brysbaert et al., described in the paper
        "Concreteness  ratings  for  40  thousand generally known english word lemmas".
    """

    def __init__(self, indent):
        super(ConcretenessCollector, self).__init__(indent)
        self.concreteness_dataset = {}

    def collect_line(self, index, line, print_info):
        """ Each line in the input file is of the following format:
            <word> <bigram indicator> <concreteness mean> ... (some other not interesting columns)
            We care about the first three columns: the word, whether it's a bigram, and what is the concreteness of the
            word.
        """

        if index == 0:
            return

        split_line = line.split()

        # The first number in each line is the bigram indicator: find it
        for token in split_line:
            if len(token) > 0 and not token[0].isalpha():
                break
        bigram_indicator = int(token)

        if bigram_indicator == 1:
            # It's a bigram: the first two tokens are the word
            word = split_line[0] + ' ' + split_line[1]
            concreteness = float(split_line[3])
        else:
            # It's a unigram: the word is only the first token
            word = split_line[0]
            concreteness = float(split_line[2])

        self.concreteness_dataset[word] = concreteness

    def file_progress_report(self, index, dataset_size, time_from_prev):
        self.log_print('Starting line ' + str(index) + ', time from previous checkpoint ' + str(time_from_prev))
