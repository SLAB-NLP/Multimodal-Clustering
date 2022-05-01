##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
from models_src.wrappers.model_wrapper import ModelWrapper
import torch
import torch.nn as nn


class BimodalClusterModelWrapper(ModelWrapper):
    """ This is the base class for bimodal cluster model wrappers.
        The visual wrapper and text wrapper will inherit from this class. """

    def __init__(self, device, config, model_dir, model_name, indent):
        super(BimodalClusterModelWrapper, self).__init__(device, config, model_dir, model_name, indent)
        self.cached_output = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.cached_loss = None

    # Abstract methods

    """Perform a single training step given the provided inputs and labels. """

    @abc.abstractmethod
    def training_step(self, inputs, labels):
        return

    """Run inference given the provided inputs. """

    @abc.abstractmethod
    def inference(self, inputs):
        return

    """ Get the relevant threshold for the model. """

    @abc.abstractmethod
    def get_threshold(self):
        return

    """ The name of the model (text/visual). """

    @abc.abstractmethod
    def get_name(self):
        return

    """ Change the underlying model to evaluation mode. """

    @abc.abstractmethod
    def eval(self):
        return

    # Implemented methods

    """ Predict a list of N bits (where N is the total number of clusters), where the ith entry is 1 iff the input to
        the last inference is associated with the ith cluster. """

    def predict_cluster_indicators(self):
        cluster_indicators = torch.zeros(self.cached_output.shape).to(self.device)
        cluster_indicators[self.cached_output >= self.get_threshold()] = 1
        return cluster_indicators

    """Same as predict_cluster_indicators but here its in the form of the list of cluster indices. """

    def predict_cluster_lists(self):
        cluster_indicators = self.predict_cluster_indicators()
        predicted_cluster_lists = [[x for x in range(self.config.cluster_num) if cluster_indicators[i, x] == 1]
                                   for i in range(cluster_indicators.shape[0])]

        return predicted_cluster_lists

    def print_info_on_loss(self):
        return self.get_name() + ' loss: ' + str(self.cached_loss)

    def print_info_on_inference(self):
        predictions_num = torch.sum(self.predict_cluster_indicators()).item()
        return 'Predicted ' + str(predictions_num) + ' clusters according to ' + self.get_name()

    # Abstract methods inherited from ModelWrapper class

    @abc.abstractmethod
    def generate_underlying_model(self):
        return

    @abc.abstractmethod
    def dump_underlying_model(self):
        return

    @abc.abstractmethod
    def load_underlying_model(self):
        return
