##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
from loggable_object import LoggableObject
import os
from utils.general_utils import project_root_dir


datasets_dir = os.path.join(project_root_dir, '..', 'datasets')


class DatasetBuilder(LoggableObject):
    """ This class is the base class for all external datasets builders.
        A dataset builder is an object that given an external dataset (e.g., an image directory and a file with matching
        captions) builds the actual dataset.
    """

    def __init__(self, indent):
        super(DatasetBuilder, self).__init__(indent)

        # The datasets are assumed to be located in a sibling directory named 'datasets'
        self.datasets_dir = datasets_dir

        # This is the directory in which we will keep the cached files of the datasets we create
        self.cached_dataset_files_dir = os.path.join(project_root_dir, 'cached_dataset_files')
        if not os.path.isdir(self.cached_dataset_files_dir):
            os.mkdir(self.cached_dataset_files_dir)

    @staticmethod
    def set_datasets_dir(dir_path):
        global datasets_dir
        datasets_dir = dir_path

    @staticmethod
    def get_datasets_dir():
        return datasets_dir

    """ Build the torch.utils.data.Data object (the actual dataset) given the configuration provided. """

    @abc.abstractmethod
    def build_dataset(self, config=None):
        return
