import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils import conv_set, branch
from backbone import *
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import time
from decimal import *
import tools

class myYOLOv1(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.45, hr=False, backbone='r18'):
        super(myYOLOv1, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32]
        if not trainable:
            self.grid_cell, self.stride_tensor = self.set_init(input_size)
            self.input_size = input_size
            self.scale = np.array([[input_size[1], input_size[0], input_size[1], input_size[0]]])
            self.scale_torch = torch.tensor(self.scale.copy()).float()

        if backbone == 'r18':
            self.backbone = resnet18(pretrained=trainable)
            C_3 = 512
            C_2 = 256
            C_1 = 128
          
        elif backbone == 'd19':
            self.backbone = darknet19(pretrained=trainable, hr=hr)
            C_3 = 1024
            C_2 = 512
            C_1 = 256

        # detection head
        self.conv_set_1 = conv_set(C_1+C_2, C_1//2, C_1, 3, leakyReLU=True)
        self.conv_set_2 = conv_set(C_2+C_3, C_2//2, C_2, 3, leakyReLU=True)
        self.conv_set_3 = conv_set(C_3, C_3//2, C_3, 3, leakyReLU=True)

        self.branch_1 = branch(C_1, leakyReLU=True)
        self.branch_2 = branch(C_2, leakyReLU=True)
        self.branch_3 = branch(C_3, leakyReLU=True)
        
        self.pred_1 = nn.Conv2d(C_1, 1 + self.num_classes + 4, 1)
        self.pred_2 = nn.Conv2d(C_2, 1 + self.num_classes + 4, 1)
        self.pred_3 = nn.Conv2d(C_3, 1 + self.num_classes + 4, 1)
    
    def set_init(self, input_size):
        total = sum([(input_size[1]//s) * (input_size[0]//s) for s in self.stride])
        grid_cell = torch.zeros(1, total, 4).to(self.device)
        stride_tensor = torch.zeros(1, total).to(self.device).float()
        start_index = 0
        for s in self.stride:
            # make a feature map size corresponding the scale
            ws = input_size[1] // s
            hs = input_size[0] // s
            for ys in range(hs):
                for xs in range(ws):
                    index = ys * ws + xs + start_index
                    grid_cell[:, index, :] = torch.tensor([xs, ys, 0, 0]).float()
                    stride_tensor[:, index] = s
            start_index += ws * hs
        
        return grid_cell, stride_tensor

    def decode_boxes(self, pred):
        """
        input box :  [delta_x, delta_y, sqrt(w), sqrt(h)]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros(pred.size())
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2])
        pred[:, :, 2:] = torch.relu(pred[:, :, 2:])
        # [delta_x, delta_y, w, h] -> [c_x, c_y, w, h]
        pred = self.grid_cell + pred
        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0]*self.stride_tensor - pred[:, :, 2]*self.input_size[1] / 2
        output[:, :, 1] = pred[:, :, 1]*self.stride_tensor - pred[:, :, 3]*self.input_size[0] / 2
        output[:, :, 2] = pred[:, :, 0]*self.stride_tensor + pred[:, :, 2]*self.input_size[1] / 2
        output[:, :, 3] = pred[:, :, 1]*self.stride_tensor + pred[:, :, 3]*self.input_size[0] / 2
        
        return output

    def clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        if boxes.shape[0] == 0:
            return boxes

        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
        """
        bbox_pred: (HxW, 4), bsize = 1
        prob_pred: (HxW, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds

    def forward(self, x):
        # backbone
        fmp_1, fmp_2, fmp_3 = self.backbone(x)

        # detection head
        # multi scale feature map fusion
        fmp_3 = self.conv_set_3(fmp_3)
        fmp_3_up = F.interpolate(fmp_3, scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = self.conv_set_2(fmp_2)
        fmp_2_up = F.interpolate(fmp_2, scale_factor=2.0, mode='bilinear', align_corners=True)
        
        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = self.conv_set_1(fmp_1)

        # s=32
        fmp_3 = self.branch_3(fmp_3)
        pred_3 = self.pred_3(fmp_3)
        B, C, _, _ = pred_3.size()
        pred_3 = pred_3.contiguous().view(B, C, -1)

        # s=16
        fmp_2 = self.branch_2(fmp_2)
        pred_2 = self.pred_2(fmp_2).contiguous().view(B, C, -1)

        # s=8
        fmp_1 = self.branch_1(fmp_1)
        pred_1 = self.pred_1(fmp_1).contiguous().view(B, C, -1)

        # [B, C, H*W] -> [B, H*W, C]
        total_prediction = torch.cat([pred_1, pred_2, pred_3], -1).permute(0, 2, 1)

        # 整理，便于训练
        if not self.trainable:
            with torch.no_grad():
                # batch size = 1
                all_obj = torch.sigmoid(total_prediction[0, :, :1])
                
                all_class = (torch.softmax(total_prediction[0, :, 1:1+self.num_classes], 1)*all_obj)
                all_local = self.decode_boxes(total_prediction[:, :, 1+self.num_classes:])[0] / self.scale_torch
                
                # # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_local = all_local.to('cpu').numpy()

                #output = self.detect(all_local, all_conf)
                bboxes, scores, cls_inds = self.postprocess(all_local, all_class)
                # clip the boxes
                bboxes *= self.scale
                bboxes = self.clip_boxes(bboxes, self.input_size) / self.scale

    
                # print(len(all_boxes))
                return bboxes, scores, cls_inds
        
        return total_prediction