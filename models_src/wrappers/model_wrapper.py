##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch
import abc
import os
from loggable_object import LoggableObject


class ModelWrapper(LoggableObject):
    """ This is the base class for all model wrappers.
    A model wrapper is a class that contains the underlying model (e.g., ResNet), is able to dump and load it, train it,
    make relevant predictions and so on. """

    def __init__(self, device, config, model_dir, model_name, indent):
        """ The init function is used both for creating a new instance (when config is specified), and for loading
        saved instances (when config is None). """
        super(ModelWrapper, self).__init__(indent)
        self.device = device  # CPU or GPU
        self.model_dir = model_dir
        self.model_name = model_name
        self.dump_path = os.path.join(self.model_dir, model_name)

        # Check if we need to load an existing model (the provided configuration is None), or create new model (the
        # provided configuration isn't None)
        need_to_load = (config is None)
        if need_to_load:
            self.load_config()
        else:
            self.config = config

        self.underlying_model = self.generate_underlying_model()

        if need_to_load:
            self.load_underlying_model()

    """Create the underlying model. """

    @abc.abstractmethod
    def generate_underlying_model(self):
        return

    """Dump the underlying model to an external file. """

    @abc.abstractmethod
    def dump_underlying_model(self):
        return

    """Load the underlying model from an external file. """

    @abc.abstractmethod
    def load_underlying_model(self):
        return

    """Get the prefix of the path to which the model and configuration will be dumped. """

    def get_dump_path(self):
        return self.dump_path

    """Get the path to which the configuration will be dumped. """

    def get_config_path(self):
        return self.get_dump_path() + '.cfg'

    """Get the path to which the underlying_model will be dumped. """

    def get_underlying_model_path(self):
        return self.get_dump_path() + '.mdl'

    """Dump the configuration to an external file. """

    def dump_config(self):
        torch.save(self.config, self.get_config_path())

    """Load the configuration from an external file. """

    def load_config(self):
        config_path = self.get_config_path()
        if not os.path.isfile(config_path):
            self.log_print('Couldn\'t find model "' + str(self.model_name) + '" in directory ' + str(self.model_dir))
            assert False
        self.config = torch.load(config_path)

    """Dump underlying model and configuration to an external file.
    If a suffix is provided, add it to the end of the name of the dumped file. """

    def dump(self, suffix=None):
        old_dump_path = self.dump_path
        if suffix is not None:
            self.dump_path += '_' + suffix

        self.dump_config()
        self.dump_underlying_model()

        self.dump_path = old_dump_path

