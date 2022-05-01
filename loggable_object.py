##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
from utils.general_utils import log_print


class LoggableObject:
    """ The top class from which all other classes inherit.
    Enables printing to the global log (using the log_print) function, with indent. The indent is stored in the
    instance, and can be incremented or decremented. """

    def __init__(self, indent):
        self.indent = indent
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def increment_indent(self):
        self.indent += 1

    def decrement_indent(self):
        self.indent -= 1

    def log_print(self, my_str):
        log_print(self.__class__.__name__, self.indent, my_str)
