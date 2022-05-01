##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

import os
import csv
from utils.general_utils import visual_dir, text_dir, default_model_name
from executors.trainers.trainer import Trainer
from executors.evaluators.common_text_evaluator import CommonTextEvaluator
from models_src.wrappers.visual_model_wrapper import VisualModelWrapper
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper


class MultimodalClusteringModelTrainer(Trainer):
    """ This class trains the main model studied by our work: a self-supervised multimodal clustering model. """

    def __init__(self, model_root_dir, training_set, epoch_num, config, token_count, indent,
                 loaded_model_dir=None, loaded_model_name=None):
        """
        model_root_dir: The directory in which the trained models, the logs, and the results csv will be saved.
        config: Configuration of the model (ClusterModelConfig object).
        token_count: The token count in the training set.
        loaded_model_dir/name: In case we continue training an existing model, this is the directory in which the model
        is located, and its name.
        """
        super().__init__(training_set, epoch_num, 50, indent)

        self.model_root_dir = model_root_dir
        if loaded_model_dir is None:
            # We train a brand new model
            self.visual_model_dir = os.path.join(model_root_dir, visual_dir)
            self.text_model_dir = os.path.join(model_root_dir, text_dir)
            os.mkdir(self.visual_model_dir)
            os.mkdir(self.text_model_dir)
            self.model_name = default_model_name
            config_for_loading = config
        else:
            # We continue training an existing model
            self.visual_model_dir = os.path.join(loaded_model_dir, visual_dir)
            self.text_model_dir = os.path.join(loaded_model_dir, text_dir)
            self.model_name = loaded_model_name
            config_for_loading = None
        self.visual_model = VisualModelWrapper(self.device, config_for_loading, self.visual_model_dir,
                                               self.model_name, indent + 1)
        self.text_model = TextCountsModelWrapper(self.device, config_for_loading, self.text_model_dir,
                                                 self.model_name, indent + 1)

        self.visual_loss_history = []
        self.text_loss_history = []

        self.prev_checkpoint_batch_ind = 0
        self.token_count = token_count
        self.evaluation_results = []

    # Inherited methods

    def dump_models(self, suffix=None):
        self.visual_model.dump(suffix)
        self.text_model.dump(suffix)

    """ Actions that should be performed at the beginning of training. """

    def pre_training(self):
        self.dump_models()

    """ Actions that should be performed at the end of training. """

    def post_training(self):
        self.dump_models()

    """ Actions performed after every epoch: we dump the models, evaluate them, and if they are the best so far, we dump
        again as 'best_model'.
    """

    def post_epoch(self):
        self.dump_models()

        self.log_print('Evaluating after finishing the epoch...')
        # If test data was provided, we evaluate after every epoch
        self.evaluate_current_model()

        # Dump results into a csv file
        self.dump_results_to_csv()

        # If this is the best model, save it
        self.dump_best_model_if_needed()

    def train_on_batch(self, index, sampled_batch, print_info):
        # Load data
        image_tensor = sampled_batch['image'].to(self.device)
        captions = sampled_batch['caption']
        token_lists = self.training_set.prepare_data(captions)

        # Infer
        self.visual_model.inference(image_tensor)
        self.text_model.inference(token_lists)
        if print_info:
            self.log_print(self.visual_model.print_info_on_inference())
            self.log_print(self.text_model.print_info_on_inference())

        # Train text model, assuming that the visual model is already trained
        # 1. Use visual model for inference
        labels_by_visual = self.visual_model.predict_cluster_indicators()

        # 2. Use the result to train textual model
        self.text_model.training_step(token_lists, labels_by_visual)

        # Train visual model, assuming that the text model is already trained
        # 1. Use textual model for inference
        labels_by_text = self.text_model.predict_cluster_indicators()

        # 2. Use the result to train visual model
        self.visual_model.training_step(image_tensor, labels_by_text)

        # Document loss
        loss = self.text_model.cached_loss + self.visual_model.cached_loss
        self.loss_history.append(loss)
        self.text_loss_history.append(self.text_model.cached_loss)
        self.visual_loss_history.append(self.visual_model.cached_loss)
        if print_info:
            batch_count_from_prev_checkpoint = len(self.loss_history) - self.prev_checkpoint_batch_ind
            mean_loss = \
                sum(self.loss_history[self.prev_checkpoint_batch_ind:]) / batch_count_from_prev_checkpoint
            mean_text_loss = \
                sum(self.text_loss_history[self.prev_checkpoint_batch_ind:]) / batch_count_from_prev_checkpoint
            mean_visual_loss = \
                sum(self.visual_loss_history[self.prev_checkpoint_batch_ind:]) / batch_count_from_prev_checkpoint
            self.log_print('Mean loss: ' + str(mean_loss) +
                           ', mean text loss: ' + str(mean_text_loss) +
                           ', mean visual loss: ' + str(mean_visual_loss))
            self.prev_checkpoint_batch_ind = len(self.loss_history)

    # Current class specific functionality

    """ Check if the current model is the best model so far in the defined metric (FScore of word clustering),
        and if so- dump it as the best model.
    """

    def dump_best_model_if_needed(self):
        all_res = [x['include_unknown_FScore'] for x in self.evaluation_results]
        if all_res[-1] == max(all_res):
            # Current model is the best model so far
            self.dump_models('best')

    """ Create a dumpable dictionary mapping metric to the results of the metric after each training epoch.
        The format of the dictionary is designed to fit the results csv, and is the following:
        {
            'metric1' : {'metric': 'metric1', '0': '3.5', '1': '3.6', ...},
            'metric2' : {'metric': 'metric2', '0': '(1,2)', '1': '(1,3)', ...}
        }
    """

    def generate_metric_to_results_mapping(self):
        metric_to_results = {}
        precision_str = "%.4f"
        for epoch_ind in range(len(self.evaluation_results)):
            epoch_results = self.evaluation_results[epoch_ind]
            for metric_name, metric_result in epoch_results.items():
                if metric_name not in metric_to_results:
                    metric_to_results[metric_name] = {'metric': metric_name}
                if isinstance(metric_result, float) or isinstance(metric_result, int):
                    result_str = precision_str % metric_result
                elif isinstance(metric_result, tuple):
                    result_str = str([precision_str % x for x in metric_result])
                else:
                    # Not implemented
                    assert False
                metric_to_results[metric_name][str(epoch_ind)] = result_str

        return metric_to_results

    """ Run evaluation metric on current model (this function will be called after every epoch). """

    def evaluate_current_model(self):
        evaluator = CommonTextEvaluator(self.text_model_dir, self.model_name, self.token_count, self.indent + 1)
        results = evaluator.evaluate()
        self.evaluation_results.append(results)

    """ Dump evaluation results to external csv file. """

    def dump_results_to_csv(self):
        csv_filename = 'evaluation_by_epoch.csv'

        with open(os.path.join(self.model_root_dir, csv_filename), 'w', newline='') as csvfile:
            step_num = len(self.evaluation_results)
            fieldnames = ['metric'] + [str(x) for x in range(step_num)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            metric_to_results = self.generate_metric_to_results_mapping()
            for result_dic in metric_to_results.values():
                writer.writerow(result_dic)
