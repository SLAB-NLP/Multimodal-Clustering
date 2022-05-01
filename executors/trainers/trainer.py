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


class Trainer(Executor):
    """ This class is the base class for all model trainers. """

    def __init__(self, training_set, epoch_num, batch_size, indent, shuffle=True):
        super().__init__(indent)

        self.epoch_num = epoch_num
        self.training_set = training_set
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.loss_history = []

    # Abstract methods, should be implemented by inheritors

    """ Dump models' current state. """

    @abc.abstractmethod
    def dump_models(self, suffix=None):
        return

    """ Actions that should be performed at the beginning of training. """

    @abc.abstractmethod
    def pre_training(self):
        return

    """ Actions that should be performed at the end of training. """

    @abc.abstractmethod
    def post_training(self):
        return

    """ Actions that should be performed at the end of each training epoch. """

    @abc.abstractmethod
    def post_epoch(self):
        return

    """ The core function: train the model given a batch of samples. """

    @abc.abstractmethod
    def train_on_batch(self, index, sampled_batch, print_info):
        return

    # Implemented methods

    """ Train on the training set; This should be the entry point of this class. """

    def train(self):
        self.pre_training()

        for epoch_ind in range(self.epoch_num):
            self.log_print('Starting epoch ' + str(epoch_ind))

            dataloader = data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=self.shuffle)

            checkpoint_len = 1000
            self.increment_indent()
            for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                                  self.train_on_batch, self.progress_report)
            self.decrement_indent()

            self.post_epoch()

        self.post_training()
