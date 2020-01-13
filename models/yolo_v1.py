import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils import *
from backbone import *
import numpy as np
import tools

class myYOLOv1(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.45, hr=False, backbone='r18'):
        super(myYOLOv1, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32
        if not trainable:
            self.grid_cell = self.set_init(input_size)
            self.input_size = input_size
            self.scale = np.array([[input_size[1], input_size[0], input_size[1], input_size[0]]])
            self.scale_torch = torch.tensor(self.scale.copy()).float()

        # we use resnet as backbone
        if backbone == 'r18':
            self.backbone = resnet18(pretrained=True)
            ch = 512
            m_ch = 256
  
        elif backbone == 'r50':
            self.backbone = resnet50(pretrained=True)
            ch = 2048
            m_ch = 512
        
        elif backbone == 'd19':
            self.backbone = darknet19(pretrained=True)
            ch = 1024
            m_ch = 512

        self.conv_set = conv_set(ch, m_ch, ch, 3, leakyReLU=True)
        self.branch = branch(ch, leakyReLU=True)
        self.pred = nn.Conv2d(ch, 1 + self.num_classes + 4, 1)
    
    def set_init(self, input_size):
        s = self.stride
        ws = input_size[1] // s
        hs = input_size[0] // s
        total = ws * hs
        grid_cell = torch.zeros(1, total, 4).to(self.device)
        
        # make a feature map size corresponding the scale
        for ys in range(hs):
            for xs in range(ws):
                index = ys * ws + xs
                grid_cell[:, index, :] = torch.tensor([xs, ys, 0, 0]).float()
        
        return grid_cell

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
        output[:, :, 0] = pred[:, :, 0]*self.stride - pred[:, :, 2]*self.input_size[1] / 2
        output[:, :, 1] = pred[:, :, 1]*self.stride - pred[:, :, 3]*self.input_size[0] / 2
        output[:, :, 2] = pred[:, :, 0]*self.stride + pred[:, :, 2]*self.input_size[1] / 2
        output[:, :, 3] = pred[:, :, 1]*self.stride + pred[:, :, 3]*self.input_size[0] / 2
        
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
        _, _, C_5 = self.backbone(x)

        # head
        C = self.conv_set(C_5)
        C = self.branch(C)
        prediction = self.pred(C)
        prediction = prediction.view(prediction.shape[0], prediction.shape[1], -1).permute(0, 2, 1)

        # 整理，便于训练
        if not self.trainable:
            with torch.no_grad():
                # batch size = 1
                all_obj = torch.sigmoid(prediction[0, :, :1])
                # mask_obj = all_obj > self.obj_thresh
                all_class = (torch.softmax(prediction[0, :, 1:1+self.num_classes], 1)*all_obj)
                all_local = self.decode_boxes(prediction[:, :, 1+self.num_classes:])[0] / self.scale_torch
                
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
        
        return prediction