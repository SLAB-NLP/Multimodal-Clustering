##########################################
### A Computational Acquisition Model  ###
### for Multimodal Word Categorization ###
##########################################
# Written by Uri Berger, December 2021.
#
# COMMERCIAL USE AND DISTRIBUTION OF THIS CODE, AND ITS MODIFICATIONS,
# ARE PERMITTED ONLY UNDER A COMMERCIAL LICENSE FROM THE AUTHOR'S EMPLOYER.

from metrics.sensitivity_specificity_metrics.compare_to_gt_bbox_metrics.compare_to_gt_bbox_metric \
    import CompareToGTBBoxMetric


class HeatmapMetric(CompareToGTBBoxMetric):
    """ This metric measures whether the maximum-valued pixel of a predicted heatmap is inside the gt bounding box. """

    def __init__(self, visual_model):
        super(HeatmapMetric, self).__init__(visual_model)

    def match_pred_to_bbox(self, sample_predicted_heatmap_centers, sample_gt_bboxes):
        gt_bbox_num = len(sample_gt_bboxes)
        heatmap_to_matching_bbox = {}
        for predicted_heatmap_ind in range(len(sample_predicted_heatmap_centers)):
            heatmap_to_matching_bbox[predicted_heatmap_ind] = []
            max_heatmap_loc = sample_predicted_heatmap_centers[predicted_heatmap_ind]

            for gt_bbox_ind in range(gt_bbox_num):
                gt_bbox = sample_gt_bboxes[gt_bbox_ind]
                # Check if maximum valued pixel is inside the gt bounding box
                if gt_bbox[0] <= max_heatmap_loc[0] <= gt_bbox[2] and \
                        gt_bbox[1] <= max_heatmap_loc[1] <= gt_bbox[3]:
                    heatmap_to_matching_bbox[predicted_heatmap_ind].append(gt_bbox_ind)

        return heatmap_to_matching_bbox

    def predict(self):
        return self.visual_model.predict_activation_map_centers()

    def get_name(self):
        return 'heatmap prediction'
