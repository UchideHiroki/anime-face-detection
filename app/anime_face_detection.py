import numpy as np
import cv2
from faster_rcnn_wrapper import FasterRCNNSlim
import tensorflow as tf
import argparse
import os
import json
import time
from nms_wrapper import NMSType, NMSWrapper


class AnimeFaceDetection:
    """
    顔検知を行うためのクラス
    """
    
    def __init__(self):        
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        sess = tf.Session(config=cfg)
        self.sess = sess

        net = FasterRCNNSlim()
        saver = tf.train.Saver()
        saver.restore(sess, "../model/res101_faster_rcnn_iter_60000.ckpt")
        self.net = net
    
    def detect(self, image):
        # pre-processing image for Faster-RCNN
        img_origin = image.astype(np.float32, copy=True)
        img_origin -= np.array([[[102.9801, 115.9465, 112.7717]]])
    
        img_shape = img_origin.shape
        img_size_min = np.min(img_shape[:2])
        img_size_max = np.max(img_shape[:2])
    
        img_scale = 600 / img_size_min
        if np.round(img_scale * img_size_max) > 1000:
            img_scale = 1000 / img_size_max
        img = cv2.resize(img_origin, None, None, img_scale, img_scale, cv2.INTER_LINEAR)
        img_info = np.array([img.shape[0], img.shape[1], img_scale], dtype=np.float32)
        img = np.expand_dims(img, 0)
    
        # test image
        _, scores, bbox_pred, rois = self.net.test_image(self.sess, img, img_info)
    
        # bbox transform
        boxes = rois[:, 1:] / img_scale
    
        boxes = boxes.astype(bbox_pred.dtype, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1
        heights = boxes[:, 3] - boxes[:, 1] + 1
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        dx = bbox_pred[:, 0::4]
        dy = bbox_pred[:, 1::4]
        dw = bbox_pred[:, 2::4]
        dh = bbox_pred[:, 3::4]
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]
        pred_boxes = np.zeros_like(bbox_pred, dtype=bbox_pred.dtype)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        # clipping edge
        pred_boxes[:, 0::4] = np.maximum(pred_boxes[:, 0::4], 0)
        pred_boxes[:, 1::4] = np.maximum(pred_boxes[:, 1::4], 0)
        pred_boxes[:, 2::4] = np.minimum(pred_boxes[:, 2::4], img_shape[1] - 1)
        pred_boxes[:, 3::4] = np.minimum(pred_boxes[:, 3::4], img_shape[0] - 1)
        return scores, pred_boxes