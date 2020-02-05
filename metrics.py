import numpy as np
from sklearn.metrics import confusion_matrix

class custom_conf_matrix():
    def __init__(self, lbl,n_class):
        self.lbl = lbl
        self.n_class = n_class
        self.conf_mat = np.zeros((self.n_class, self.n_class))
    def update_step(self,truth_lbl,pred_lbl):
        if (truth_lbl == 255).all():
            return
        
        curr_conf_mat = confusion_matrix(y_true=truth_lbl,
                                                    y_pred=pred_lbl,
                                                    labels=self.lbl)
        
#        if self.conf_mat is not None:
            
        self.conf_mat += curr_conf_mat
        #else:
        #    self.overall_confusion_matrix = current_confusion_matrix     

    def compute_mean_iou(self):
        
        intersection = np.diag(self.conf_mat)
        ground_truth_set = self.conf_mat.sum(axis=1)
        predicted_set = self.conf_mat.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        return np.mean(intersection / union.astype(np.float32))
        #mean_intersection_over_union = np.mean(intersection_over_union)
        
        #return mean_intersection_over_union
    def reset(self):
        self.conf_mat = np.zeros((self.n_class, self.n_class))
