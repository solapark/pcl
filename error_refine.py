import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
from morph_filter import *


class ERROR_REFINE :
    def __init__(self, pcl_normal, pcl_src, quant_axis, quant_interval, margin, refine_kind, is_show_image, wait_key, refine_path, kernel_size=0):
        self.pcl_normal = pcl_normal
        self.pcl_src = pcl_src
        self.pcl_dst = np.array([], dtype='f')
        self.quant_axis = quant_axis
        self.quant_interval = quant_interval
        self.margin = margin
        self.refine_kind = refine_kind
        self.is_show_image = is_show_image
        self.wait_key = wait_key
        self.refine_path = refine_path
        if(kernel_size) : self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.bins, self.bin_size = self.get_quant_bins()
        self.quant_inds = self.get_quant_inds()
 
    def refine(self) :
        for i in np.arange(self.bin_size) :
            src_img, axis0_offset, axis1_offset = self.get_quant_img(i)
            if(self.refine_kind == "CLOSE") :
                dst_img = cv2.morphologyEx(src_img,cv2.MORPH_CLOSE, self.kernel) 
            elif(self.refine_kind == "OPEN") :
                dst_img = cv2.morphologyEx(src_img,cv2.MORPH_OPEN, self.kernel) 
            elif(self.refine_kind == "BRIDGE") :
                dst_img = morph_bridge(src_img)

            if(self.is_show_image) : 
                self.show_image(src_img, dst_img)

            self.refine_pcl(dst_img, i, axis0_offset, axis1_offset)
    def show_image(self, src, dst):
        '''
        cv2.imshow("src", src)
        cv2.imshow("dst", dst)
        cv2.waitKey(self.wait_key)       
        '''
        concat_img = np.concatenate((src, dst), axis = 1)
        concat_img = cv2.resize(concat_img, dsize=(0,0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.refine_kind, concat_img)
        cv2.waitKey(self.wait_key)       
        
    def get_quant_bins(self):
        x = self.pcl_src[:, self.quant_axis]    
        x_min = x.min() + 1
        x_max = x.max() + self.quant_interval
        bins = np.arange(start=x_min, stop=x_max, step=self.quant_interval) 
        bin_size = len(bins)
        return bins, bin_size

    def get_quant_inds(self):
        x = self.pcl_src[:, self.quant_axis]    
        inds = np.digitize(x, self.bins)
        return inds

    def get_quant_img(self, idx):
        target_inds = (self.quant_inds == idx)
        target_p = self.pcl_src[target_inds]
        img = np.delete(target_p, self.quant_axis, 1)
        img_axis0 = img[:, 0] 
        img_axis1 = img[:, 1] 

        img_axis0_size = img_axis0.max() + margin
        img_axis1_size = img_axis1.max() + margin
        black = np.zeros(shape = (img_axis0_size, img_axis1_size))
        black[img_axis0, img_axis1] = 1
        img_axis0_start = img_axis0.min() - margin
        img_axis1_start = img_axis1.min() - margin
        black_final = black[img_axis0_start:, img_axis1_start:]
        return black_final, img_axis0_start, img_axis1_start

    def refine_pcl(self, img, bin_idx, axis0_offset, axis1_offset):
        #quant_value = self.bins[self.quant_inds[bin_idx]] -1
        quant_value = self.bins[bin_idx] -1
        dst_coord_raw = np.where(img > 0) 
        dst_coord = (dst_coord_raw[0] + axis0_offset, dst_coord_raw[1] + axis1_offset)
        dst_coord = np.vstack((dst_coord[0], dst_coord[1]))
        dst_coord = np.swapaxes(dst_coord, 0, 1)
        self.dst_coord_final = np.insert(dst_coord, self.quant_axis, quant_value, axis=1)
        self.pcl_dst = np.vstack( (self.pcl_dst, self.dst_coord_final)) if self.pcl_dst.any() else self.dst_coord_final

    def save(self) :
        cloud = pcl.PointCloud()
        self.pcl_dst = self.pcl_dst.astype(np.float32)
        cloud.from_array(self.pcl_dst)
        pcl.save(cloud, self.refine_path, format="ply")

if __name__=="__main__":
    pcl_path ="PointCloud_Web/longdress.ply"
    error_path ="PointCloud_Web/longdress_add_error.ply"
#error_path ="PointCloud_Web/longdress_del_20.ply"
    refine_path = "PointCloud_Web/longdress_add_error_refine.ply"
    quant_interval = 1
    quant_axis = 1
    margin = 10
    refine_kind = "BRIDGE"
    is_show_image =1 
    wait_key = 0
    kernel_size=2

    p = pcl.load(pcl_path)
    p_np = np.asarray(p).astype(int)
    p_error = pcl.load(error_path)
    p_error_np = np.asarray(p_error).astype(int)

    error_refine = ERROR_REFINE(pcl_path, p_error_np, quant_axis, quant_interval, margin, refine_kind, is_show_image, wait_key, refine_path, kernel_size)
    error_refine.refine()
    error_refine.save()
