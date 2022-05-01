##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
from dataset_builders.dataset_builder import DatasetBuilder


class ClassificationDatasetBuilder(DatasetBuilder):
    """ This class is the base class for all classification datasets: a dataset where samples contain class annotations.
    """

    def __init__(self, indent):
        super(ClassificationDatasetBuilder, self).__init__(indent)

    """ Get a mapping from class index to class name, for classes labeled in the dataset. """

    @abc.abstractmethod
    def get_class_mapping(self):
        return
