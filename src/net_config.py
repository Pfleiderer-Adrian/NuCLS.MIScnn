import os
import random
from datetime import datetime
from tensorboard import program


class net_config:

    def __init__(self, data_path_img, data_path_mask, classes, raw_classes, batch_size, epochs, analysis, loss, info, train_data_split = 0.7, val_data_split = 0.15, pred_data_split = 0.15):
        self.data_path_img = data_path_img
        self.data_path_mask = data_path_mask
        self.classes = classes
        self.raw_classes = raw_classes
        self.exclude_classes = [x for x in raw_classes if x not in classes]
        self.train_data_split = train_data_split
        self.val_data_split = val_data_split
        self.pred_data_split = pred_data_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.analysis = analysis
        self.info = info
        self.loss = loss



