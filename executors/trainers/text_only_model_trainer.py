##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import torch.utils.data as data
from executors.trainers.trainer import Trainer
from models_src.wrappers.text_model_wrapper import TextOnlyCountsModelWrapper
from utils.general_utils import default_model_name, for_loop_with_reports
from dataset_builders.build_word_list_for_categorization import build_word_list_for_categorization


class TextOnlyModelTrainer(Trainer):
    """ This class trains the text-only baseline model, based on word co-occurrence counts. """

    def __init__(self, model_root_dir, training_set, config, indent):
        """
        model_root_dir: The directory in which the trained models, the logs, and the results csv will be saved.
        config: Configuration of the model (TextOnlyModelConfig object).
        """
        super().__init__(training_set, 1, 1, indent)

        self.model = TextOnlyCountsModelWrapper(self.device, config, model_root_dir, default_model_name, indent + 1)

    # Inherited methods

    def dump_models(self, suffix=None):
        self.model.dump(suffix)

    """ Before starting, we need to go over the training set and build the vocabulary. """

    def pre_training(self):
        dataloader = data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=self.shuffle)

        self.log_print('Going over training set to build vocabulary...')
        checkpoint_len = 10000
        self.increment_indent()
        for_loop_with_reports(dataloader, len(dataloader), checkpoint_len,
                              self.train_on_batch, self.progress_report)
        self.decrement_indent()
        self.log_print('Vocabulary built!')

        self.model.stop_building_vocab()

    def post_training(self):
        token_count = self.training_set.get_token_count()

        word_list = build_word_list_for_categorization(self.indent + 1)

        cluster_num = self.model.config.cluster_num
        self.model.word_to_cluster = self.model.underlying_model.categorize_words(word_list, cluster_num)

        # Create concreteness prediction
        self.model.concreteness_prediction = self.model.underlying_model.predict_word_concreteness(token_count)

        self.dump_models()

    def post_epoch(self):
        return

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        captions = sampled_batch['caption']
        token_lists = self.training_set.prepare_data(captions)

        # Update counts
        self.model.training_step(token_lists, None)
