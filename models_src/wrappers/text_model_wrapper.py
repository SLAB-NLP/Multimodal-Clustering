##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import abc
import random
import torch

from utils.general_utils import models_dir

from models_src.wrappers.bimodal_cluster_model_wrapper import BimodalClusterModelWrapper
from models_src.wrappers.word_clustering_model_wrapper import WordClusteringModelWrapper
from models_src.wrappers.concreteness_prediction_model_wrapper import ConcretenessPredictionModelWrapper

from models_src.underlying_models.word_co_occurrence_model import WordCoOccurrenceModel
from models_src.underlying_models.word_cluster_count_model import WordClusterCountModel


class TextModelWrapper(BimodalClusterModelWrapper, WordClusteringModelWrapper, ConcretenessPredictionModelWrapper):
    """ This class wraps the text underlying model.
    It contains functionality shared by all text wrappers, different from the visual wrappers in the fact that we need
    to predict clusters for each word first (before we predict for the entire sentences). """

    def __init__(self, device, config, model_dir, model_name, indent):
        super(TextModelWrapper, self).__init__(device, config, model_dir, model_name, indent)

    # Abstract methods

    """ Predict all associated cluster for a given word.
        Unlike predict_cluster_for_word, we don't only return the maximum probability cluster, but all the clusters
        with probability that exceeds the threshold.
        If the word is unknown, return None. """

    @abc.abstractmethod
    def predict_clusters_for_word(self, word):
        return

    # Implemented methods

    """ Get the relevant threshold for the model (inherited from parent class). """

    def get_threshold(self):
        return self.config.text_threshold

    """ For each word in the input, predict if it is associated with any cluster with a probability higher than the
        threshold."""

    def predict_cluster_associated_words(self, sentences):
        res = []
        for sentence in sentences:
            res.append([])
            for token in sentence:
                cluster_instantiating_token = False
                while True:
                    prediction_res = self.predict_cluster_for_word(token)
                    if prediction_res is None:
                        # Never seen this token before
                        break
                    predicted_cluster, prob = prediction_res
                    if prob >= self.get_threshold():
                        cluster_instantiating_token = True
                    break
                if cluster_instantiating_token:
                    res[-1].append(1)
                else:
                    res[-1].append(0)

        return res

    """ Create a mapping between our induced clusters and the ground-truth classes, in the following way:
        First, map each gt class to a single cluster (or to no cluster) by providing its name as input to the model.
        Next, map each cluster to the list of gt classes that were mapped to it. """

    def create_cluster_to_gt_class_mapping(self, class_mapping):
        gt_class_to_prediction = {i: self.predict_cluster_for_word(class_mapping[i])
                                  for i in class_mapping.keys()
                                  if ' ' not in class_mapping[i]}
        gt_class_to_cluster = {x[0]: x[1][0] for x in gt_class_to_prediction.items() if x[1] is not None}
        cluster_num = self.config.cluster_num
        cluster_to_gt_class = {cluster_ind: [] for cluster_ind in range(cluster_num)}
        for gt_class_ind, cluster_ind in gt_class_to_cluster.items():
            cluster_to_gt_class[cluster_ind].append(gt_class_ind)

        return cluster_to_gt_class

    @staticmethod
    def get_name():
        return 'text'

    def print_info_on_inference(self):
        predictions_num = torch.sum(self.predict_cluster_indicators()).item()
        return 'Predicted ' + str(predictions_num) + ' clusters according to text'


class TextCountsModelWrapper(TextModelWrapper):
    """ In this class, the underlying model is a count model, that predicts using statistics on the counts of
        words-cluster co-occurrence. """

    def __init__(self, device, config, model_dir, model_name, indent):
        super().__init__(device, config, model_dir, model_name, indent)
        self.underlying_model.calculate_probs()

    def generate_underlying_model(self):
        return WordClusterCountModel(self.config.cluster_num, self.config.text_underlying_model)

    def eval(self):
        self.underlying_model.calculate_probs()

    def training_step(self, inputs, labels):
        loss = self.criterion(self.cached_output, labels)
        loss_val = loss.item()
        self.cached_loss = loss_val

        batch_size = len(inputs)

        for caption_ind in range(batch_size):
            predicted_clusters_by_image = [x for x in range(self.config.cluster_num)
                                           if labels[caption_ind, x] == 1]
            for token in inputs[caption_ind]:
                for cluster_ind in predicted_clusters_by_image:
                    self.underlying_model.document_co_occurrence(token, cluster_ind)

    def inference(self, inputs):
        self.underlying_model.calculate_probs()
        batch_size = len(inputs)

        with torch.no_grad():
            output_tensor = torch.zeros(batch_size, self.config.cluster_num).to(self.device)
            for caption_ind in range(batch_size):
                for token in inputs[caption_ind]:
                    prediction_res = self.predict_cluster_for_word(token)
                    if prediction_res is None:
                        # Never seen this token before
                        continue
                    predicted_cluster, prob = prediction_res
                    if output_tensor[caption_ind, predicted_cluster] < prob:
                        output_tensor[caption_ind, predicted_cluster] = prob

        self.cached_output = output_tensor
        return output_tensor

    def dump_underlying_model(self):
        torch.save(self.underlying_model, self.get_underlying_model_path())

    def load_underlying_model(self):
        self.underlying_model = torch.load(self.get_underlying_model_path())

    def predict_cluster_for_word(self, word):
        return self.underlying_model.predict_cluster(word)

    def predict_clusters_for_word(self, word):
        cluster_conditioned_on_word = self.underlying_model.get_cluster_conditioned_on_word(word)
        if cluster_conditioned_on_word is None:
            # return [0]*self.config.cluster_num
            return None

        cluster_indicators = [1 if x >= self.get_threshold() else 0 for x in cluster_conditioned_on_word]
        return cluster_indicators

    """ For all input tokens, return the highest probability that it's associated with any cluster.
        For unknown words, return 0. """

    def estimate_word_concreteness(self, sentences):
        res = []
        for sentence in sentences:
            res.append([])
            for token in sentence:
                prediction_res = self.predict_cluster_for_word(token)
                if prediction_res is None:
                    # Never seen this token before
                    res[-1].append(0)
                else:
                    predicted_cluster, prob = prediction_res
                    res[-1].append(prob)

        return res


class TextOnlyCountsModelWrapper(TextModelWrapper):
    """ This class will wrap a text-only co-occurrence model. This differs from TextCountsModelWrapper in the fact
        that TextCountsModelWrapper receives external supervision (from its visual counterpart), while this model only
        uses words co-occurrences in the training set sentences. """

    def __init__(self, device, config, model_dir, model_name, indent):
        super(TextOnlyCountsModelWrapper, self).__init__(device, config, model_dir, model_name, indent)

        # Training mode: 0 means we're building the vocab, 1 means we're collecting co-occurrences
        self.training_mode = 0

        if config is not None:
            # This is a new model
            self.word_to_cluster = None
            self.concreteness_prediction = None

    def stop_building_vocab(self):
        self.training_mode = 1
        self.underlying_model.create_co_occurrence_matrix()

    def generate_underlying_model(self):
        return WordCoOccurrenceModel()

    def training_step(self, inputs, labels):
        batch_size = len(inputs)

        for caption_ind in range(batch_size):
            if self.training_mode == 0:
                for token in inputs[caption_ind]:
                    self.underlying_model.document_word(token)
            else:
                cur_input = inputs[caption_ind]
                token_num = len(cur_input)
                for token_ind in range(token_num):
                    token = cur_input[token_ind]
                    self.underlying_model.document_word_occurrence(token)
                    for other_token_ind in range(token_ind + 1, token_num):
                        other_token = cur_input[other_token_ind]
                        self.underlying_model.document_co_occurrence(token, other_token)

    def inference(self, inputs):
        return

    def eval(self):
        return

    def dump_underlying_model(self):
        torch.save([self.underlying_model, self.word_to_cluster, self.concreteness_prediction],
                   self.get_underlying_model_path())

    def load_underlying_model(self):
        self.underlying_model, self.word_to_cluster, self.concreteness_prediction = \
            torch.load(self.get_underlying_model_path())

    def predict_cluster_for_word(self, word):
        if word in self.word_to_cluster:
            return self.word_to_cluster[word], 1
        else:
            return None

    def predict_clusters_for_word(self, word):
        res = [0]*self.config.cluster_num
        prediction = self.predict_cluster_for_word(word)
        if prediction is not None:
            res[prediction[0]] = 1
        return res

    def estimate_word_concreteness(self, sentences):
        res = []
        for sentence in sentences:
            res.append([])
            for token in sentence:
                if token in self.concreteness_prediction:
                    res[-1].append(self.concreteness_prediction[token])
                else:
                    # Never seen this token before
                    res[-1].append(0)

        return res

    def predict_cluster_associated_words(self, sentences):
        return


class TextRandomModelWrapper(TextModelWrapper):
    """ This class is a model that makes random predictions (needed as a baseline). """

    def __init__(self, device, config, indent):
        super(TextRandomModelWrapper, self).__init__(device, config, models_dir, 'random', indent)

    def training_step(self, inputs, labels):
        return

    def inference(self, inputs):
        return

    def predict_cluster_for_word(self, word):
        return random.randrange(self.config.cluster_num), 1

    def predict_clusters_for_word(self, word):
        return

    def estimate_word_concreteness(self, sentences):
        return

    def generate_underlying_model(self):
        return

    def dump_underlying_model(self):
        return

    def load_underlying_model(self):
        return

    def eval(self):
        return
