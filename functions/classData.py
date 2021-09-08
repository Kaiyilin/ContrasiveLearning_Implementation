import os, sys, math, scipy
import numpy as np 
import nibabel as nib
from tensorflow.python.ops.gen_math_ops import ceil

class Data(object):
    def __init__(self) -> None:
        pass


    # Data Preprocessing
    def normalised(image):
        image = (image - image.min())/(image.max() - image.min()) 
        return image

    def standardised(image): 
        img = (image - image.mean())/image.std() 
        return img 
    
    def padding_func(array, pad_size):
        # define padding size
        elements = array.shape    
        for element in elements:
            if element > pad_size:
                sys.exit('\nThe expanded dimension shall be greater than your current dimension')
        pad_list = list() 
        for i in range(array.ndim):
            x = pad_size - array.shape[i]
            if x%2 ==1:
                y_1 = (x/2 +0.5)
                y_2 = (x/2 -0.5)
                z = (int(y_1),int(y_2))
                pad_list.append(z)

            else:
                y = int(x/2)
                z=(y,y)
                pad_list.append(z)
        pad_array = np.pad(array, pad_list, 'constant')
        pad_list = list() 
        return pad_array

    def split(c,array):
        array_val = array[:c,:,:,:]
        array_tr = array[c:,:,:,:]
        return array_tr, array_val
    # Read files
    def get_image_data(dirr, pad_func= False,  pre_func = None):
        """
        This version can import 3D array regardless of the size
        """
        #cwd = os.getcwd()
        array_list = []
        imgs_array = np.array([])
        path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
        path_list.sort() #對讀取的路徑進行排序
        #print(path_list)
        for file in path_list:
            if file.endswith(".nii"):
                img = nib.load(os.path.join(dirr, file))
                img_array = img.get_fdata()
                if pre_func:
                    img_array = pre_func(img_array)
                
                if pad_func:
                    pad_size = math.ceil(math.log(max(img_array.shape), 2))
                    img_array = Data.padding_func(img_array, pad_size= pad_size)
                    
                img_array = img_array[None,...]
                array_list.append(img_array)
            else:
                pass 
        imgs_array = np.concatenate(array_list, axis=0)

        return imgs_array

class Augmentation(Data):
    def __init__(self) -> None:
        super().__init__()
        pass
            
    # Augmentation
    def translateit(image, offset, isseg=False):
        order = 0 if isseg == True else 5

        return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')

    def translateit2(image, offset, isseg=False):
        order = 0 if isseg == True else 5

        return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), int(offset[2])), order=order, mode='nearest')

    def rotateit_y(image, theta, isseg=False):
        order = 0 if isseg == True else 5
            
        return scipy.ndimage.rotate(image, float(theta), axes=(0,2), reshape=False, order=order, mode='nearest') # Shall detemined reshape or not, also rotate toward which axis?

    def rotateit_x(image, theta, isseg=False):
        order = 0 if isseg == True else 5
            
        return scipy.ndimage.rotate(image, float(theta), axes=(1,2), reshape=False, order=order, mode='nearest') # Shall detemined reshape or not, also rotate toward which axis?

    def rotateit_z(image, theta, isseg=False):
        order = 0 if isseg == True else 5
            
        return scipy.ndimage.rotate(image, float(theta), axes=(0,1), reshape=False, order=order, mode='nearest') # Shall detemined reshape or not, also rotate toward which axis?

class Padding(Data):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data


    def padding_func(array, pad_size):
        # define padding size
        elements = array.shape    
        for element in elements:
            if element > pad_size:
                sys.exit('\nThe expanded dimension shall be greater than your current dimension')
        pad_list = list() 
        for i in range(array.ndim):
            x = pad_size - array.shape[i]
            if x%2 ==1:
                y_1 = (x/2 +0.5)
                y_2 = (x/2 -0.5)
                z = (int(y_1),int(y_2))
                pad_list.append(z)

            else:
                y = int(x/2)
                z=(y,y)
                pad_list.append(z)
        pad_array = np.pad(array, pad_list, 'constant')
        pad_list = list() 
        return pad_array
    