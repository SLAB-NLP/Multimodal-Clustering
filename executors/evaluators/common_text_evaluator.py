##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from executors.executor import Executor

# Metrics
from metrics.categorization_metric import CategorizationMetric
from metrics.concreteness_prediction_metric import ConcretenessPredictionMetric
from metrics.cluster_counter_metric import ClusterCounterMetric
from metrics.word_association_metric import WordAssociationMetric

# Datasets
from dataset_builders.category_dataset import CategoryDatasetBuilder
from dataset_builders.concreteness_dataset import ConcretenessDatasetBuilder
from dataset_builders.swow_dataset import SWOWDatasetBuilder
from dataset_builders.build_word_list_for_categorization import build_word_list_for_categorization

# Models
from models_src.wrappers.text_model_wrapper import TextCountsModelWrapper


class CommonTextEvaluator(Executor):
    """ This evaluator is the most commonly used in our project (after each epoch in the training of multimodal
        clustering model). It evaluates the model on text tasks.
    """

    def __init__(self, text_model_dir, model_name, token_count, indent):
        super().__init__(indent)

        self.token_count = token_count

        # Load Model
        self.model = TextCountsModelWrapper(self.device, None, text_model_dir, model_name, indent + 1)
        self.model.eval()

        # Datasets
        word_list = build_word_list_for_categorization(self.indent + 1)
        word_dict = {x: True for x in word_list}
        category_dataset = CategoryDatasetBuilder(self.indent + 1).build_dataset()
        filtered_category_dataset = {x[0]: [y for y in x[1] if y in word_dict] for x in category_dataset.items()}
        concreteness_dataset = ConcretenessDatasetBuilder(self.indent + 1).build_dataset()
        swow_dataset = SWOWDatasetBuilder(word_list, self.indent + 1).build_dataset()

        self.metrics = [
            CategorizationMetric(self.model, filtered_category_dataset, ignore_unknown_words=True),
            CategorizationMetric(self.model, category_dataset, ignore_unknown_words=False),
            ConcretenessPredictionMetric(self.model, concreteness_dataset, token_count),
            ClusterCounterMetric(self.model, token_count),
            WordAssociationMetric(self.model, word_list, swow_dataset)
        ]

    """ The entry point of this class. """

    def evaluate(self):
        results = {}
        for metric in self.metrics:
            self.log_print(metric.report())
            results.update(metric.results)
        self.decrement_indent()

        return results
