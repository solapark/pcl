from scipy import ndimage
import numpy as np

def morph_bridge(img) :
    filter_list = list() 
    #filter_list.append(np.array([[1,0,1], [0,0,0], [0,0,0]]))
    filter_list.append(np.array([[1,-1,-1], [-1,0,1], [0,0,0]]))
    #filter_list.append(np.array([[1,0,0], [0,0,0], [1,0,0]]))
    filter_list.append(np.array([[1,-1,0], [-1,0,0], [-1,1,0]]))
    filter_list.append(np.array([[1,0,0], [0,0,0], [0,0,1]]))
    #filter_list.append(np.array([[0,1,0], [1,0,0], [0,0,0]]))
    #filter_list.append(np.array([[0,1,0], [0,0,1], [0,0,0]]))
    filter_list.append(np.array([[-1,1,0], [-1,0,0], [1,-1,0]]))
    filter_list.append(np.array([[0,1,0], [0,0,0], [0,1,0]]))
    filter_list.append(np.array([[0,1,-1], [0,0,-1], [0,-1,1]]))
    filter_list.append(np.array([[-1,-1,1], [1,0,0], [0,0,0]]))
    filter_list.append(np.array([[0,0,1], [0,0,0], [1,0,0]]))
    filter_list.append(np.array([[0,-1,1], [0,0,-1], [0,1,-1]]))
    #filter_list.append(np.array([[0,0,1], [0,0,0], [0,0,1]]))
    filter_list.append(np.array([[0,0,0], [1,0,1], [0,0,0]]))
    #filter_list.append(np.array([[0,0,0], [1,0,0], [0,1,0]]))
    filter_list.append(np.array([[0,0,0], [1,0,-1], [-1,-1,1]]))
    filter_list.append(np.array([[0,0,0], [-1,0,1], [1,-1,-1]]))
    #filter_list.append(np.array([[0,0,0], [0,0,1], [0,1,0]]))
    #filter_list.append(np.array([[0,0,0], [0,0,0], [1,0,1]]))

    new_image = np.copy(img)
    for conv in filter_list :
        new_image += (ndimage.convolve(img, conv, mode = 'constant', cval=0.0) > 1)

    return new_image

if __name__ == "__main__" :
    a = np.array([[1,0,0],[1,0,1],[0,0,1]])
    new_image =  morph_bridge(a)

    print('old_image')
    print(a)
    print('new_image')
    print(new_image)
