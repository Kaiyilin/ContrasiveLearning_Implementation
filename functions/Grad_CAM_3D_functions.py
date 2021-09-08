from __future__ import absolute_import, division, print_function, unicode_literals
from functions.All_functions import *

def Grad_CAM(input_dir,background_dir,output_dir,labels,input_size,model_path,pad_size):
    """
    can I use tree?
    background_dir: The directory of your background, better using a mask
    output_dir: Where you'd like to save those Grad_CAM images
    labels: same as the labels you given in your training procedure, this mainly helps you to identify which data were wrong
    pad_size: default is None
    """
    def readfile_pad_for_overlap(dirr, pad_size):
        os.chdir(dirr)
        cwd = os.getcwd()
        for root, dirs, files in os.walk(cwd):
            for file in files:
                if file.endswith(".nii"):
                #print(os.path.join(root, file))
                    img = nib.load(os.path.join(root, file))
                    img_array = img.get_fdata()
                    img_array = tf.keras.utils.normalize(img_array)
                    img_array = padding_zeros(img_array, pad_size)
        return img_array

    _,imgs_tensor, imgs_list = myreadfile_pad(input_dir, pad_size)
    imgs_tensor = imgs_tensor[...,None]
    model=load_model(model_path)
    #model.summary()
    conv_layer_list = list([])
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:#check for convolutional layer
            continue
        #print(i, layer.name, layer.output.shape)#summarize output shape
        conv_layer_list.append(i)
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=conv_layer_list[-1]).output, model.output])
    #grad_model.summary()

    incorrect_list = list([])
    index_init = 0
    for img_tensor in imgs_tensor:
        with tf.GradientTape() as tape:
            #Compute GRADIENT
            img_tensor_2 = img_tensor[None,...]
            conv_outputs, predictions = grad_model(img_tensor_2)
            class_index=int(tf.math.argmax(predictions,axis=1))
            loss = predictions[:, class_index]
        if class_index != labels:
            incorrect_list.append(imgs_list[index_init])

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        # Average gradients spatially
        weights = tf.reduce_mean(grads, axis=(0,1,2))
        # Build a ponderated map of filters according to gradients importance
        cam = np.zeros(output.shape[0:3], dtype=np.float64)

        for index, w in enumerate(weights):
            cam += w * output[:, :, :, index]

        from skimage.transform import resize
        from matplotlib import pyplot as plt
        capi=resize(cam,(pad_size,pad_size,pad_size))
        capi = np.maximum(capi,0)
        heatmap = (capi - capi.min()) / (capi.max() - capi.min())
        f, axarr = plt.subplots(8,8,figsize=(15,10))
        f.suptitle('Grad-CAM') 

        background = readfile_pad_for_overlap(background_dir,64) 
        os.chdir(output_dir)
        import math
        #sag
        for slice_count in range(pad_size):
            axial_img=background[slice_count,:,:]
            axial_grad_cmap_img=heatmap[slice_count,:,:]
            axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(pad_size), 0),round(math.sqrt(pad_size), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
        plt.savefig('Grad_CAM_sag_%s.png'%(imgs_list[index_init]))
        plt.close()

        #cor
        for slice_count in range(pad_size):
            axial_img=background[:,slice_count,:]
            axial_grad_cmap_img=heatmap[:,slice_count,:]
            axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(pad_size), 0),round(math.sqrt(pad_size), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
        plt.savefig('Grad_CAM_cor_%s.png'%(imgs_list[index_init]))
        plt.close()
        #axial
        for slice_count in range(pad_size):
            axial_img=background[:,:,slice_count]
            axial_grad_cmap_img=heatmap[:,:,slice_count]
            axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(pad_size), 0),round(math.sqrt(pad_size), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
        plt.savefig('Grad_CAM_axial_%s.png'%(imgs_list[index_init]))
        plt.close()

        incorrect_list_df = pd.DataFrame(incorrect_list)
        incorrect_list_df.to_csv('Incorrect_list.csv')
        index_init+=1

def Grad_CAM_function(images, labels, model, save_path):
    from nilearn.datasets import load_mni152_template
    from skimage.transform import resize
    pad_size = images.shape[1]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.chdir(save_path)

    # Generate grad model
    conv_layer_list = list([])
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:#check for convolutional layer
            continue
        conv_layer_list.append(i)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=conv_layer_list[-1]).output, model.output])

    # Compute gradient

    for i in range(len(images)):
        with tf.GradientTape() as tape:
            #Compute GRADIENT
            image = images[i]
            image = image[None,...]
            image = tf.cast(image, tf.float32)
            conv_outputs, predictions = grad_model(image)

            class_index=int(tf.math.argmax(predictions,axis=1))

            if class_index == labels[i]:
                loss = predictions[:, class_index]
                # Extract filters and gradients
                output = conv_outputs[0]
                #print(output.shape)
                grads = tape.gradient(loss, conv_outputs)[0]
                #print(grads.shape)
                # Average gradients spatially
                weights = tf.reduce_mean(grads, axis=(0,1,2))
                print(weights)
                # Build a ponderated map of filters according to gradients importance
                cam = np.zeros(output.shape[0:3], dtype=np.float32)

                # resize the CAM
                for index, w in enumerate(weights):
                    cam += w * output[:, :, :, index]
                capi=resize(cam,(pad_size,pad_size,pad_size))
                capi = np.maximum(capi,0)
                heatmap = (capi - capi.min()) / (capi.max() - capi.min())
                f, axarr = plt.subplots(10,10,figsize=(15,10))
                f.suptitle('Grad-CAM') 
            # Generate a background image
                background = load_mni152_template()
                background = background.get_fdata()
                background = padding_zeros(background, pad_size)

                for slice_count in range(pad_size):
                    axial_img=np.squeeze(background[:,:,slice_count])
                    axial_grad_cmap_img=np.squeeze(heatmap[:,:,slice_count])
                    axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.7, 0, dtype = cv2.CV_32F)
                    plt.subplot(8,16,slice_count+1)
                    plt.imshow(axial_overlay,cmap='jet')
                    plt.axis('off')
                plt.savefig('Grad_CAM_axial_%s.png'%(i))
                plt.close()

#Grad_CAM_function(val_images,val_labels,model, Grad_CAM_save_path)


def Grad_CAM_2(images, labels, model, save_path):
    from nilearn.datasets import load_mni152_template
    from skimage.transform import resize
    pad_size = images.shape[1]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.chdir(save_path)

    # Generate grad model
    conv_layer_list = list([])
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if len(layer.output.shape) != 5:#check for convolutional layer
            continue
        conv_layer_list.append((i,layer.output.shape))
    print(conv_layer_list)
    selected_index = input('Select the layer:')
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=int(selected_index)).output, model.output])

    # Compute gradient

    for i in range(len(images)):
        with tf.GradientTape() as tape:
            #Compute GRADIENT
            image = images[i]
            image = image[None,...]
            image = tf.cast(image, tf.float32)
            conv_outputs, predictions = grad_model(image)

            class_index=int(tf.math.argmax(predictions,axis=1))

            if class_index == labels[i]:
                loss = predictions[:, class_index]
                # Extract filters and gradients
                output = conv_outputs[0]
                #print(output.shape)
                grads = tape.gradient(loss, conv_outputs)[0]
                #print(grads.shape)
                # Average gradients spatially
                weights = tf.reduce_mean(grads, axis=(0,1,2))
                print(weights)
                # Build a ponderated map of filters according to gradients importance
                cam = np.zeros(output.shape[0:3], dtype=np.float32)

                # resize the CAM
                for index, w in enumerate(weights):
                    cam += w * output[:, :, :, index]
                capi=resize(cam,(pad_size,pad_size,pad_size))
                capi = np.maximum(capi,0)
                heatmap = (capi - capi.min()) / (capi.max() - capi.min())
                f, axarr = plt.subplots(10,10,figsize=(15,10))
                f.suptitle('Grad-CAM') 
            # Generate a background image
                background = load_mni152_template()
                background = background.get_fdata()
                background = padding_zeros(background, pad_size)

                for slice_count in range(pad_size):
                    axial_img=np.squeeze(background[:,:,slice_count])
                    axial_grad_cmap_img=np.squeeze(heatmap[:,:,slice_count])
                    axial_overlay=cv2.addWeighted(axial_img,0.3,axial_grad_cmap_img, 0.7, 0, dtype = cv2.CV_32F)
                    plt.subplot(8,16,slice_count+1)
                    plt.imshow(axial_overlay,cmap='jet')
                    plt.axis('off')
                plt.savefig('Grad_CAM_axial_%s.png'%(i))
                plt.close()



def Grad_CAM_3(image_tensors, background, save_path, model):
    """
    background_dir: The directory of your background, better using a mask
    output_dir: Where you'd like to save those Grad_CAM images
    labels: same as the labels you given in your training procedure, this mainly helps you to identify which data were wrong
    pad_size: default is None
    """
    shape = image_tensors[0,:,:,:,0].shape
    model = model
    conv_layer_list = list([])
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:
            continue
        conv_layer_list.append(i)
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=conv_layer_list[-1]).output, model.output])

    index_init = 1
    for img_tensor in image_tensors:
        with tf.GradientTape() as tape:
            #Compute GRADIENT
            img_tensor_2 = img_tensor[None,...]
            conv_outputs, predictions = grad_model(img_tensor_2)
            class_index=(tf.argmax(predictions[0]))
            loss = predictions[:, class_index]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        # Average gradients spatially
        weights = tf.reduce_mean(grads, axis=(0,1,2))
        # Build a ponderated map of filters according to gradients importance
        cam = np.zeros(output.shape[0:3], dtype=np.float64)

        for index, w in enumerate(weights):
            cam += w * output[:, :, :, index]

        from skimage.transform import resize
        from matplotlib import pyplot as plt
        capi=resize(cam, shape)
        capi = np.maximum(capi,0)
        heatmap = (capi - capi.min()) / (capi.max() - capi.min())
        f, axarr = plt.subplots(8,8,figsize=(15,10))
        f.suptitle('Grad-CAM') 

        os.chdir(save_path)
        import math

        #sag
        for slice_count in range(shape[0]):
            axial_img = background[slice_count,:,:]
            axial_grad_cmap_img=heatmap[slice_count,:,:]
            axial_overlay=cv2.addWeighted(axial_img,0.3 ,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(shape[0]), 0),round(math.sqrt(shape[0]), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
            plt.axis('off')
        plt.savefig(f'{index_init}_Grad_CAM_sag.png')
        plt.close()

        #cor
        for slice_count in range(shape[0]):
            axial_img = background[:,slice_count,:]
            axial_grad_cmap_img=heatmap[:,slice_count,:]
            axial_overlay=cv2.addWeighted(axial_img,0.3 ,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(shape[0]), 0),round(math.sqrt(shape[0]), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
            plt.axis('off')
        plt.savefig(f'{index_init}_Grad_CAM_cor.png')
        plt.close()
        #axial
        for slice_count in range(shape[0]):
            axial_img = background[:,:,slice_count]
            axial_grad_cmap_img=heatmap[:,:,slice_count]
            axial_overlay=cv2.addWeighted(axial_img, 0.3,axial_grad_cmap_img, 0.6, 0, dtype = cv2.CV_32F)
            plt.subplot(round(math.sqrt(shape[0]), 0),round(math.sqrt(shape[0]), 0),slice_count+1)
            plt.imshow(axial_overlay,cmap='jet')
            plt.axis('off')
        plt.savefig(f'{index_init}_Grad_CAM_axial.png')
        plt.close()
        index_init += 1