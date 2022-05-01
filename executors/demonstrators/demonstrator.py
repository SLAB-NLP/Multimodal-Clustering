##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch.utils.data as data
from utils.general_utils import for_loop_with_reports
from executors.executor import Executor
import abc


class Demonstrator(Executor):
    """ This class is the base class for all demonstrators: classes that are meant to demonstrate some aspect of
    the dataset. """

    def __init__(self, dataset, num_of_items_to_demonstrate, indent):
        super().__init__(indent)

        self.dataset = dataset
        self.num_of_items_to_demonstrate = num_of_items_to_demonstrate

    @abc.abstractmethod
    def demonstrate_item(self, index, sampled_batch, print_info):
        return

    def demonstrate(self):
        data_loader = data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        for_loop_with_reports(data_loader, len(data_loader), len(data_loader),
                              self.demonstrate_item, self.progress_report)
