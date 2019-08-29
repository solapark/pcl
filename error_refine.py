import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
from morph_filter import *

pcl_path ="PointCloud_Web/longdress.ply"
error_path ="PointCloud_Web/longdress_add_error.ply"
#error_path ="PointCloud_Web/longdress_del_20.ply"
quant_interval = 1
quant_axis = 1
margin = 10

def get_quant_bins(pcl_org, axis, interval):
    x = pcl_org[:, axis]    
    x_min = x.min() +1
    x_max = x.max() + interval
    bins = np.arange(start=x_min, stop=x_max, step=interval) 
    bin_size = len(bins)
    return bins, bin_size

def get_quant_inds(point, bins, axis):
    x = point[:, axis]  
    inds = np.digitize(x, bins)
    return inds

def get_quant_img(pcl_org, quant_inds, idx, margin, quant_axis):
    target_inds = (quant_inds == idx)
    target_p = pcl_org[target_inds]
    img = np.delete(target_p, quant_axis, 1)
    img_axis_a = img[:, 0] 
    img_axis_b = img[:, 1] 

    img_a_size = img_axis_a.max() + margin
    img_b_size = img_axis_b.max() + margin
    black = np.zeros(shape = (img_a_size, img_b_size))
    black[img_axis_a, img_axis_b] = 1
    img_a_start = img_axis_a.min() - margin
    img_b_start = img_axis_b.min() - margin
    black_final = black[img_a_start:, img_b_start:]
    return black_final

if __name__=="__main__":
    p = pcl.load(pcl_path)
    p_np = np.asarray(p).astype(int)
    p_error = pcl.load(error_path)
    p_error_np = np.asarray(p_error).astype(int)

    bins, bin_size = get_quant_bins(p_np, quant_axis, quant_interval)
    print('bin size', bin_size)
    quant_inds_org = get_quant_inds(p_np, bins, quant_axis)
    quant_inds_error = get_quant_inds(p_error_np, bins, quant_axis)

    for i in np.arange(bin_size) :
        quant_org_img = get_quant_img(p_np, quant_inds_org, i, margin, quant_axis)
        quant_error_img = get_quant_img(p_error_np, quant_inds_error, i, margin, quant_axis)
        kernel = np.ones((2, 2), np.uint8)
        morp_bridge = morph_bridge(quant_error_img)
        morp_bridge_erosion = cv2.erode(morp_bridge, kernel, iterations=1)
        morp_open = cv2.morphologyEx(quant_error_img,cv2.MORPH_OPEN, kernel) 
        morp_open_bridge = morph_bridge(morp_open)
        morp_closing = cv2.morphologyEx(quant_error_img,cv2.MORPH_CLOSE, kernel) 
        morp_open_close = cv2.morphologyEx(morp_open,cv2.MORPH_CLOSE, kernel) 
        dilate_image = cv2.dilate(quant_error_img, kernel) 

        cv2.imshow("org", quant_org_img)
        cv2.imshow("error", quant_error_img)
 
        cv2.imshow("morp_open", morp_open)
        cv2.imshow("morp_open_bridge", morp_open_bridge)
        cv2.imshow("morp_open_close", morp_open_close)
        #cv2.imshow("morp_closing", morp_closing)
        #cv2.imshow("morp_bridge", morp_bridge)
        #cv2.imshow("dilate", dilate_image)

        cv2.waitKey(0)
