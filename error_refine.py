import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
from morph_filter import *

pcl_path ="PointCloud_Web/longdress.ply"
error_path ="PointCloud_Web/longdress_error_refine/longdress_del_20.ply"
refine_path = "PointCloud_Web/longdress_error_refine/longdress_del_20_bridge_color.ply"
quant_interval = 1
quant_axis = 1
margin = 10
#refine_kind = ["CLOSE", "BRIDGE"]
#refine_kind = ["CLOSE"]
refine_kind = ["BRIDGE"]
#refine_kind = ["EROSION"]
is_show_image =0
wait_key = 0
kernel_size=2

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
        self.quant_inds = self.get_quant_inds(self.pcl_src)
        self.quant_inds_normal = self.get_quant_inds(self.pcl_normal)
 
    def refine(self) :
        for i in np.arange(self.bin_size) :
            src_img, color_img, axis0_offset, axis1_offset, axis0_size, axis1_size = self.get_quant_img(i, self.pcl_src, self.quant_inds)

            dst_img_list = []
            title = "org_error"  
            for refine_kind in self.refine_kind : 
                if(refine_kind == "CLOSE") :
                    dst_img = cv2.morphologyEx(src_img,cv2.MORPH_CLOSE, self.kernel) 
                elif(refine_kind == "OPEN") :
                    dst_img = cv2.morphologyEx(src_img,cv2.MORPH_OPEN, self.kernel) 
                elif(refine_kind == "EROSION") :
                    dst_img = cv2.erode(src_img, self.kernel) 
                elif(refine_kind == "BRIDGE") :
                    dst_img = morph_bridge(src_img)
                dst_img_list.append(dst_img)
                title += "_" + refine_kind

            if(self.is_show_image) : 
                org_img, _, _, _, _, _ = self.get_quant_img(i, self.pcl_normal, self.quant_inds_normal, axis0_offset, axis1_offset, axis0_size, axis1_size)
                self.show_image(org_img, src_img, dst_img_list, title)
            color_img = self.refine_color(dst_img, color_img)
            self.refine_pcl(dst_img, color_img, i, axis0_offset, axis1_offset)
    def show_image(self, org, src, dst_img_list, title):
        concat_img = org 
        concat_img = np.concatenate((concat_img, src), axis = 1)
        for dst in dst_img_list : concat_img = np.concatenate((concat_img, dst), axis = 1) 
        concat_img = cv2.resize(concat_img, dsize=(0,0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(title, concat_img)
        cv2.waitKey(self.wait_key)       
        
    def get_quant_bins(self):
        x = self.pcl_src[:, self.quant_axis]    
        x_min = x.min() + 1
        x_max = x.max() + self.quant_interval
        bins = np.arange(start=x_min, stop=x_max, step=self.quant_interval) 
        bin_size = len(bins)
        return bins, bin_size

    def get_quant_inds(self, pcl):
        x = pcl[:, self.quant_axis]    
        inds = np.digitize(x, self.bins)
        return inds

    def get_quant_img(self, idx, pcl, quant_inds, axis0_offset=0, axis1_offset=0, axis0_size=0, axis1_size=0):
        target_inds = (quant_inds == idx)
        target_p = pcl[target_inds]
        img = np.delete(target_p, self.quant_axis, 1)
        img_axis0 = img[:, 0] 
        img_axis1 = img[:, 1] 
        img_color = img[:, -1]

        img_axis0_size = axis0_size if axis0_size else img_axis0.max() + margin
        img_axis1_size = axis1_size if axis1_size else img_axis1.max() + margin
        black = np.zeros(shape = (img_axis0_size, img_axis1_size))
        color = np.zeros(shape = (img_axis0_size, img_axis1_size))
        black[img_axis0, img_axis1] = 1
        color[img_axis0, img_axis1] = img_color
        img_axis0_start = axis0_offset if axis0_offset else img_axis0.min() - margin
        img_axis1_start = axis1_offset if axis0_offset else img_axis1.min() - margin
        black_final = black[img_axis0_start:, img_axis1_start:]
        color_final = color[img_axis0_start:, img_axis1_start:]
        return black_final, color_final, img_axis0_start, img_axis1_start, img_axis0_size, img_axis1_size

    def refine_color(self, binary_img, color_img) :
        white_pixel = (binary_img == 1)
        no_color = (color_img == 0)
        xs, ys = np.where( white_pixel & no_color)
        new_color_img = color_img.copy()
        for x, y in zip(xs, ys) :
            distance = 0
            is_found = 0
            while not is_found : 
                distance += 1
                search_space = color_img[x-distance:x+distance+1, y-distance:y+distance+1] 
                yes_color_coord = np.nonzero(search_space)
                if(yes_color_coord[0].any()):
                    yes_color = search_space[yes_color_coord]
                    new_color_img[x, y] = yes_color[0]
                    is_found = 1
        return new_color_img

    def refine_pcl(self, img, color_img, bin_idx, axis0_offset, axis1_offset):
        #quant_value = self.bins[self.quant_inds[bin_idx]] -1
        quant_value = self.bins[bin_idx] -1
        dst_coord_raw = np.where(img > 0) 
        color_value = color_img[dst_coord_raw]
        dst_coord = (dst_coord_raw[0] + axis0_offset, dst_coord_raw[1] + axis1_offset)
        dst_coord = np.vstack((dst_coord[0], dst_coord[1]))
        dst_coord = np.swapaxes(dst_coord, 0, 1)
        dst_coord_final = np.insert(dst_coord, self.quant_axis, quant_value, axis=1)
        dst_coord_with_color = np.insert(dst_coord_final, 3, color_value, axis=1)
        self.pcl_dst = np.vstack( (self.pcl_dst, dst_coord_with_color)) if self.pcl_dst.any() else dst_coord_with_color

    def save(self) :
        cloud = pcl.PointCloud_PointXYZRGBA()
        self.pcl_dst = self.pcl_dst.astype(np.float32)
        cloud.from_array(self.pcl_dst)
        pcl.save(cloud, self.refine_path, format="ply")

if __name__=="__main__":
    p = pcl.load_XYZRGBA(pcl_path)
    p_np = p.to_array().astype(int)
    p_error = pcl.load_XYZRGBA(error_path)
    p_error_np = p_error.to_array().astype(int)

    error_refine = ERROR_REFINE(p_np, p_error_np, quant_axis, quant_interval, margin, refine_kind, is_show_image, wait_key, refine_path, kernel_size)
    error_refine.refine()
    error_refine.save()
