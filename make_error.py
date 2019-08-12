import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pcl_path ="PointCloud_Web/longdress.ply"
error_pcl_path ="PointCloud_Web/longdress_add_error.ply"

is_add = 1
rand_add_lowest = -1
rand_add_highest = 1

is_del = 0
delete_prob = .2

def add_error(pcl_org, rand_lowest, rand_highest ):
	pcl_size = pcl_org.shape
	rand = np.random.randint(rand_lowest, rand_highest+1, size=pcl_size) #(p_np.shape)
	noise = pcl_org + rand
	noise = noise.astype(np.float32)
	new_pcl = np.append(pcl_org, noise, axis=0)
	return new_pcl

def del_error(pcl_org, del_prob):
	pcl_num = len(pcl_org)
	remain_size = pcl_num - int(del_prob * pcl_num)
	remain_idx = np.random.choice(pcl_num, size=remain_size, replace=False)
	new_pcl = pcl_org[remain_idx, :] 
	return new_pcl

if __name__=="__main__":
	p = pcl.load(pcl_path)
	p_np = np.asarray(p)
	if is_add :
		new_pcl = add_error(p_np, rand_add_lowest, rand_add_highest ) 
	elif is_del :
		new_pcl = del_error(p_np, delete_prob) 
	print('pcl', p_np.shape)
	print('new_pcl', new_pcl.shape)

	cloud = pcl.PointCloud()
	cloud.from_array(new_pcl)
	pcl.save(cloud, error_pcl_path, format="ply")
