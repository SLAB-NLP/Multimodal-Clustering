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


class VisualClusterClassificationMetric(VisualClassificationMetric):
    """ This metric is only for models that cluster images. The clusters are mapped to the ground-truth classes for
        evaluation using different methods. """

    def __init__(self, visual_model, class_num, mapping_mode):
        super(VisualClusterClassificationMetric, self).__init__(visual_model, class_num)

        # Maintain a list of pairs of (predicted clusters, gt classes) for future calculations
        self.predicted_clusters_gt_classes = []
        ''' Mapping mode: Two possible heuristics for choosing cluster to class mapping:
        First, for each cluster choose the class with which it co-occurred the most.
        Second, for each cluster choose the class with which it has the largest IoU. '''
        self.mapping_mode = mapping_mode

    def predict_and_document(self, visual_metadata, visual_inputs, text_inputs):
        predicted_clusters = self.visual_model.predict_cluster_lists()
        self.document(predicted_clusters, visual_metadata['gt_classes'])

    def document(self, predicted_classes, gt_classes):
        batch_size = len(gt_classes)
        self.predicted_clusters_gt_classes += \
            [(predicted_classes[i], gt_classes[i]) for i in range(batch_size)]

    def evaluate(self):
        # Get a unique list of clusters
        clusters_with_repetition = [x[0] for x in self.predicted_clusters_gt_classes]
        cluster_list = list(set([inner for outer in clusters_with_repetition for inner in outer]))

        # First, document for each cluster how many times each class co-occurred with it
        cluster_class_co_occur, predicted_cluster_count, gt_class_count = self.aggregate_co_occurrence(cluster_list)

        ''' Two possible heuristics for choosing cluster to class mapping:
        First, for each cluster choose the class with which it co-occurred the most.
        Second, for each cluster choose the class with which it has the largest IoU. '''
        if self.mapping_mode == 'co_occur':
            # First option: for each cluster, choose the class with which it co-occurred the most
            cluster_to_class = self.create_cluster_to_class_mapping_by_co_occur(cluster_class_co_occur, cluster_list)
        elif self.mapping_mode == 'iou':
            # Second option: for each cluster choose the class with which it has the largest IoU
            cluster_to_class = self.create_cluster_to_class_mapping_by_iou(cluster_class_co_occur,
                                                                           predicted_cluster_count,
                                                                           gt_class_count,
                                                                           cluster_list)

        # Finally, go over the results again and use the mapping to evaluate
        for predicted_clusters, gt_classes in self.predicted_clusters_gt_classes:
            predicted_classes = [cluster_to_class[x] for x in predicted_clusters]
            self.evaluate_classification([predicted_classes], [gt_classes])

    """ When we traversed the test set we documented co-occurrence of cluster and ground-truth classes.
        We now want a summary of co-occurrence counts.
    """

    def aggregate_co_occurrence(self, cluster_list):
        cluster_class_co_occur = {x: {} for x in cluster_list}

        predicted_cluster_count = {x: 0 for x in cluster_list}
        gt_class_count = {}
        for predicted_clusters, gt_classes in self.predicted_clusters_gt_classes:
            for predicted_cluster in predicted_clusters:
                # Increment count
                predicted_cluster_count[predicted_cluster] += 1
                # Go over gt classes
                for gt_class in gt_classes:
                    # Increment count
                    if gt_class not in gt_class_count:
                        gt_class_count[gt_class] = 0
                    gt_class_count[gt_class] += 1
                    # Document co-occurrence
                    if gt_class not in cluster_class_co_occur[predicted_cluster]:
                        cluster_class_co_occur[predicted_cluster][gt_class] = 0
                    cluster_class_co_occur[predicted_cluster][gt_class] += 1

        return cluster_class_co_occur, predicted_cluster_count, gt_class_count

    @staticmethod
    def create_cluster_to_class_mapping_by_co_occur(cluster_class_co_occur, cluster_list):
        return {
            x: max(cluster_class_co_occur[x], key=cluster_class_co_occur[x].get)
            if len(cluster_class_co_occur[x]) > 0 else None
            for x in cluster_list
        }

    @staticmethod
    def create_cluster_to_class_mapping_by_iou(cluster_class_co_occur, predicted_cluster_count, gt_class_count,
                                               cluster_list):
        intersections = cluster_class_co_occur
        unions = {
            cluster_ind: {
                class_ind:
                    predicted_cluster_count[cluster_ind] +  # Cluster count
                    gt_class_count[class_ind] -  # Class count
                    intersections[cluster_ind][class_ind]  # Intersection count
                    if class_ind in intersections[cluster_ind] else 0
                for class_ind in gt_class_count.keys()
            }
            for cluster_ind in cluster_list
        }
        ious = {
            cluster_ind: {
                class_ind:
                    intersections[cluster_ind][class_ind] / unions[cluster_ind][class_ind]
                    if unions[cluster_ind][class_ind] > 0 else 0
                for class_ind in gt_class_count.keys()
            }
            for cluster_ind in cluster_list
        }

        # Now, for each cluster, choose the class with which it co-occurred the most
        cluster_to_class = {
            x: max(ious[x], key=ious[x].get)
            if len(ious[x]) > 0 else None
            for x in cluster_list
        }

        return cluster_to_class

    def report(self):
        """In this metric we have post analysis, we'll do it in the report function as this function is
        executed after all calculations are done."""
        self.evaluate()

        res = self.report_with_name()

        return res

    def get_name(self):
        return 'Visual classification ' + str(self.mapping_mode) + ' mapping'
