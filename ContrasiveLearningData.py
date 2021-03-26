from functions.All_functions import *
from functions.cm_tensorboard import *


def importdata2(dirr,dirr1,dirr2,dirr3,dirr4,dirr5,pad_size=None):
    
    def myreadfile_pad(dirr, pad_size):
        #This version can import 3D array regardless of the size
    
        os.chdir(dirr)
        number = 0

        flag = True
        imgs_array = np.array([])
        path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
        path_list.sort()
        for file in path_list:
            if file.endswith(".nii"):
                #print(os.path.join(dirr, file))
                img = nib.load(os.path.join(dirr, file))
                img_array = img.get_fdata()
                img_array = data_preprocessing(img_array)
                #img_array = standardised(img_array)
                img_array = padding_zeros(img_array, pad_size)
                img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
                number += 1
                if flag == True:
                    imgs_array = img_array

                else:
                    imgs_array = np.concatenate((imgs_array, img_array), axis=0)

                flag = False
        return number, imgs_array, path_list

    if pad_size == None:
      a_num, first_mo, _ = myreadfile(dirr)
      b_num, second_mo, _ = myreadfile(dirr1)
      h_num, third_mo, _ = myreadfile(dirr2)
      
      a_num2, first_mo2, _ = myreadfile(dirr3)
      b_num2, second_mo2, _ = myreadfile(dirr4)
      h_num2, third_mo2, _ = myreadfile(dirr5)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

    else:
      #pad_size = int(input('Which size would you like? '))
      a_num, first_mo, _ = myreadfile_pad(dirr,pad_size)
      b_num, second_mo, _ = myreadfile_pad(dirr1,pad_size)
      h_num, third_mo, _ = myreadfile_pad(dirr2,pad_size)
      
      a_num2, first_mo2, _ = myreadfile_pad(dirr3,pad_size)
      b_num2, second_mo2, _ = myreadfile_pad(dirr4,pad_size)
      h_num2, third_mo2, _ = myreadfile_pad(dirr5,pad_size)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2


def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  val_pred_raw = model.predict(val_images)
  val_pred = np.argmax(val_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(val_labels, val_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def tf_random_rotate_image_xyz(image, image2):
    # 3 axes random rotation
    def rotateit_y(image):
        toggleSwitch = bool(random.getrandbits(1))

        if toggleSwitch == True:
            image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(0,2), reshape=False)
        else:
            image = image
        return image

    def rotateit_x(image):
        toggleSwitch = bool(random.getrandbits(1))

        if toggleSwitch == True:
            image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(1,2), reshape=False)     
        else:
            image = image  
        return image

    def rotateit_z(image):
        toggleSwitch = bool(random.getrandbits(1))
        
        if toggleSwitch == True:
            image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(0,1), reshape=False)
        else:
            image = image
        return image

    im_shape = image.shape
    [image,] = tf.py_function(rotateit_x, [image], [tf.float32])
    [image,] = tf.py_function(rotateit_y, [image], [tf.float32])
    [image,] = tf.py_function(rotateit_z, [image], [tf.float32])
    image.set_shape(im_shape)

    im2_shape = image2.shape
    [image2,] = tf.py_function(rotateit_x, [image2], [tf.float32])
    [image2,] = tf.py_function(rotateit_y, [image2], [tf.float32])
    [image2,] = tf.py_function(rotateit_z, [image2], [tf.float32])
    image2.set_shape(im2_shape)
    return (image, image2)

def tf_random_rotate_image(im1, im2):
    # one axes random rotation
    def random_rotate_image(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-90, 90), order = 0, reshape=False)
        return image
    
    im_shape = im1.shape
    [im1,] = tf.py_function(random_rotate_image, [im1], [tf.float32])
    im1.set_shape(im_shape)
    [im2,] = tf.py_function(random_rotate_image, [im2], [tf.float32])
    im2.set_shape(im_shape)

    images = (im1, im2)
    return images


# Load and split tr, test set
BA_alff, BB_alff, HC_alff, _, _, _= importdata2(dir['BA'], dir['BB'], dir['HC'], dir['BA2'], dir['BB2'], dir['HC2'], 64)

BA_alff_tr, BA_alff_val = split(5,BA_alff)
BB_alff_tr, BB_alff_val = split(5,BB_alff)
HC_alff_tr, HC_alff_val = split(5,HC_alff)
del BA_alff, BB_alff, HC_alff


BA_alff_tr, BA_alff_val = BA_alff_tr[...,None], BA_alff_val[...,None]
BB_alff_tr, BB_alff_val = BB_alff_tr[...,None], BB_alff_val[...,None]
HC_alff_tr, HC_alff_val = HC_alff_tr[...,None], HC_alff_val[...,None]

ds = np.concatenate([BA_alff_tr, BB_alff_tr, HC_alff_tr], axis = 0)
ds = ds.astype("float32")

ds_tr = tf.data.Dataset.from_tensor_slices((ds, ds))

