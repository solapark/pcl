import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pcl_path ="/home/sap/namuga/PointCloud_Web/longdress.ply"
new_dir ="/home/sap/namuga/PointCloud_Web/longdress_error_refine/"

is_add =1 
rand_lowest = -1
rand_highest = 1
add_prob = 0.1

is_del = 0
del_prob = .5

def get_error_path(org_path, is_add, is_del, add_prob, del_prob, new_dir) :
    name_exp = org_path.split('/')[-1]
    name, exp = name_exp.split('.')
    if(is_add) :
        add_prob_percent = int(add_prob * 100)
        new_name = name+"_add_"+str(add_prob_percent)+'.'+exp
    if(is_del) :
        del_prob_percent = int(del_prob * 100)
        new_name = name+"_del_"+str(del_prob_percent)+'.'+exp
    new_path = new_dir + new_name

    return new_path

def add_error(pcl_org, rand_lowest, rand_highest, add_prob):
    pcl_size = pcl_org.shape
    rand_int = np.random.randint(rand_lowest, rand_highest+1, size=pcl_size) #(p_np.shape)
    rand_int[:, -1] = 0 #color information
    p = [1-add_prob, add_prob]
    is_add = np.random.choice([0,1], (pcl_size[0], 1), p=p) #(p_np.shape)
    rand = rand_int * is_add
    noise = pcl_org + rand
    noise = noise.astype(np.float32)
    new_pcl = np.append(pcl_org, noise, axis=0)
    new_pcl = np.unique(new_pcl, axis=0)
    return new_pcl

def del_error(pcl_org, del_prob):
    pcl_num = len(pcl_org)
    remain_size = pcl_num - int(del_prob * pcl_num)
    remain_idx = np.random.choice(pcl_num, size=remain_size, replace=False)
    new_pcl = pcl_org[remain_idx, :] 
    return new_pcl

if __name__=="__main__":
    error_pcl_path =get_error_path(pcl_path, is_add, is_del, add_prob, del_prob, new_dir) 
     
    p = pcl.load_XYZRGBA(pcl_path)
    #p_np = np.asarray(p)
    p_np = p.to_array()
    if is_add :
        new_pcl = add_error(p_np, rand_lowest, rand_highest, add_prob) 
    elif is_del :
        new_pcl = del_error(p_np, del_prob) 
    print('pcl', p_np.shape)
    print('new_pcl', new_pcl.shape)

    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(new_pcl)
    pcl.save(cloud, error_pcl_path, format="ply")
