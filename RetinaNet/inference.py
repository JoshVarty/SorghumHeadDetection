import os
import sys
import PIL
import shutil
import pandas as pd
import numpy as np
from functools import partial
from sklearn.model_selection import KFold

import torch

import fastai
from fastai.core import is_tuple
from fastai.train import ShowGraph
from fastai.vision import Path, open_image, ImageBBox, ObjectItemList, get_transforms, bb_pad_collate, conv_layer
from fastai.vision import Learner, create_body, models, conv2d, ifnone, DatasetType, range_of, progress_bar, cnn_learner, Image
from fastai.torch_core import to_np
from fastai.vision.data import pil2tensor

from RetinaNet.object_detection_helper import process_output, nms, rescale_boxes, GeneralEnsemble
from RetinaNet.object_detection_helper import create_anchors, get_annotations_from_path
from RetinaNet.RetinaNetFocalLoss import FocalLoss
from RetinaNet.RetinaNet import RetinaNet
from RetinaNet.callbacks import BBLossMetrics, BBMetrics, PascalVOCMetric

#Helper methods for use during inference 
def get_crop_coordinates(image_height, image_width, verticalCropIndex, horizontalCropIndex, model_input_size):
    
    maxVerticalCrops = int(np.ceil(image_height / model_input_size))
    maxHorizontalCrops = int(np.ceil(image_width / model_input_size))
    
    lastValidVerticalCrop = image_height - model_input_size
    lastValidHorizontalCrop = image_width - model_input_size
    
    crop_x = (horizontalCropIndex % maxHorizontalCrops) * model_input_size
    crop_x = min(crop_x, lastValidHorizontalCrop)
    
    crop_y = (verticalCropIndex % maxVerticalCrops) * model_input_size
    crop_y = min(crop_y, lastValidVerticalCrop)
    
    return crop_y, crop_x
    

#Overrides fastai's default 'open_image' method to crop based on our crop counter
def setupNewCrop(verticalCropIndex, horizontalCropIndex, model_input_size=256):
    
    def open_image_with_specific_crop(fn, convert_mode, after_open):
        """
        Opens an image with a specific crop, based on horizontalCropIndex and verticalCropIndex
        """
        
        x = PIL.Image.open(fn)
        width, height = x.size
        
        crop_y, crop_x = get_crop_coordinates(height, width, verticalCropIndex, horizontalCropIndex, model_input_size)
        
        cropped_image = x.crop([crop_x, crop_y, crop_x + model_input_size, crop_y + model_input_size])    
        
        # standardize    
        return Image(pil2tensor(cropped_image, np.float32).div_(255))

    #Override fastai's open_image() to use our custom open_image_with_specific_crop()
    fastai.vision.data.open_image = open_image_with_specific_crop
    
def getMaxHeightAndWidth(learn, ds_type=DatasetType.Valid):
    """
    Returns the maximum height and width for a given image dataset 
    """
    dl = learn.dl(ds_type)

    maxHeight = 0
    maxWidth = 0
    for i in dl.x:
        height = i.shape[1]
        width = i.shape[2]
        
        if height > maxHeight:
            maxHeight = height
            
        if width > maxWidth:
            maxWidth = width
        
    return maxHeight, maxWidth

def get_bounding_box_predictions(learn, dataloader, anchors, original_images, verticalCropIndex, horizontalCropIndex, detect_threshold = 0.5, nms_threshold = 0.1, model_input_size=256):
    """
    Generates bounding box predictions for an entire epoch of a provided Dataloader
    """
    all_imgs = []
    all_bboxes = []
    
    batch_index = 0
    
    for img_batch, target_batch in dataloader:
    
        prediction_batch = learn.model(img_batch)
        class_pred_batch, bbox_pred_batch = prediction_batch[:2]

        for index, (img, clas_pred, bbox_pred) in enumerate(zip(img_batch, class_pred_batch, bbox_pred_batch)):
            original_image = original_images[batch_index + index]
            
            #Filter out predictions below detect_thresh
            bbox_pred, scores, preds = process_output(clas_pred, bbox_pred, anchors, detect_threshold)
            
            #If there are no bounding boxes, we're done
            if len(bbox_pred) <= 0:
                continue
                
            #Only keep most likely bounding boxes
            to_keep = nms(bbox_pred, scores, nms_threshold)
            
            if len(to_keep) <= 0:
                continue
                
            bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()

            #Change back to pixel values
            height = img.shape[1]
            width = img.shape[2]
            t_sz = torch.Tensor([height, width])[None].cpu()
            bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
            
            #Get crop location
            crop_y, crop_x = get_crop_coordinates(original_image.shape[1], original_image.shape[2], verticalCropIndex, horizontalCropIndex, model_input_size)

            # change from CTWH to TLRB
            bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2
            #Account for offset due to cropping
            bbox_pred[:, 0] = bbox_pred[:, 0] + crop_y
            bbox_pred[:, 2] = bbox_pred[:, 2] + bbox_pred[:, 0]
            bbox_pred[:, 1] = bbox_pred[:, 1] + crop_x
            bbox_pred[:, 3] = bbox_pred[:, 3] + bbox_pred[:, 1]

            all_imgs.append(original_image)
            all_bboxes.append(bbox_pred)
        
        #After completing a batch, we have to keep track the total number of images we've processed
        batch_index = batch_index + index + 1
        
    return all_imgs, all_bboxes


def custom_tta(learn, anchors, ds_type=DatasetType.Valid, model_input_size=256):
    dl = learn.dl(ds_type)
    
    maxHeight, maxWidth = getMaxHeightAndWidth(learn, ds_type)

    #Keep track of previous method for opening images
    old_open_image = fastai.vision.data.open_image
    try:
        maxNumberOfVerticalCrops = ((maxHeight - 1) // model_input_size) + 1
        maxNumberOfHorizontalCrops = ((maxWidth - 1) // model_input_size) + 1
        
        original_images = list(dl.x)
        
        for i in range(maxNumberOfVerticalCrops):
            for j in range(maxNumberOfHorizontalCrops):
                #Override fastai's open_image to crop at a specific location in the image
                setupNewCrop(i, j)
                
                #yield get_preds(learn.model, dl, activ=_loss_func2activ(learn.loss_func))[0]
                yield get_bounding_box_predictions(learn, dl, anchors, original_images, i, j)
    finally:
        #Restore original method for opening images
        fastai.vision.data.open_image = old_open_image
    