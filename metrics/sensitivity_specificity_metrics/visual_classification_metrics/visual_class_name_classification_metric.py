##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from metrics.sensitivity_specificity_metrics.visual_classification_metrics.visual_classification_metric \
    import VisualClassificationMetric


class VisualClassNameClassificationMetric(VisualClassificationMetric):
    """ This metric uses the name of ground-truth classes for zero-shot visual classification. """

    def __init__(self, visual_classifier, class_num):
        super(VisualClassNameClassificationMetric, self).__init__(None, class_num)
        self.classifier = visual_classifier

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_classes = self.classifier.classify_using_inferred_results()
        self.document(predicted_classes, visual_metadata['gt_classes'])

    def document(self, predicted_classes, gt_classes):
        self.evaluate_classification(predicted_classes, gt_classes)

    def report(self):
        return self.report_with_name()

    def get_name(self):
        return 'Class name classification'
