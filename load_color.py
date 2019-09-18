import pcl
import numpy as np

#src_path ="PointCloud_Web/tmp.ply"
src_path ="PointCloud_Web/test.ply"
dst_path = "PointCloud_Web/test2.ply"

if __name__=="__main__":
    p = pcl.load_XYZRGBA(src_path)
    #p_np = np.asarray(p)
    p_np = p.to_array()
    
    #new_cloud = pcl.PointCloud()
    #new_cloud = pcl.PointCloud_PointXYZRGB()
    new_cloud = pcl.PointCloud_PointXYZRGBA()
    new_cloud.from_array(p_np)
    pcl.save(new_cloud, dst_path, format="ply")
    
