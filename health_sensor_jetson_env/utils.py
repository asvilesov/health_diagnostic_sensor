#utils
import numpy as np

def calc_area(coords):
    max = 0
    temp = 0
    index = 0
    for coord, j in zip(coords, range(coords.shape[0])):
        x1, y1, x2, y2 = coord
        temp = np.abs((x2-x1)*(y2-y1))
        if(temp > max):
            max = temp
            index = j
    return max, j

def calc_center(coord):
    x1, y1, x2, y2 = coord

    x = (x2-x1)/2 + x1
    y = (y2-y1)/2 + y1

    return x,y

def calc_square_box(coord, img_dim):

    x_center, y_center = calc_center(coord)

    x1, y1, x2, y2 = coord

    xlen = np.abs(x2-x1)
    ylen = np.abs(y2-y1)

    new_len = xlen if xlen > ylen else ylen
    new_len = int(new_len*1.4)

    #Clip in case face is too far out of the camera dims
    x1 = np.clip(1 * new_len / 2 + x_center, a_min = 0, a_max = img_dim[1]-1) 
    x2 = np.clip(-1 * new_len / 2 + x_center, a_min = 0, a_max = img_dim[1]-1)
    y1 = np.clip(1 * new_len / 2 + y_center, a_min = 0, a_max = img_dim[0]-1)
    y2 = np.clip(-1 * new_len / 2 + y_center, a_min = 0, a_max = img_dim[0]-1)

    # print(new_len)
    # print(x1,x2,y1,y2)
    # print(x2-x1)
    # print(y2-y1)

    return np.array([x1,y1,x2,y2], dtype=np.long)


