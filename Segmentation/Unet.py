
import os
import matplotlib
import matplotlib.pyplot as plt

from Segmentation.Plots import *
import numpy as np
import tensorflow as tf
from PIL import ImageOps,Image
#import tensorflow_datasets as tfds
from pillow_heif import register_heif_opener
from tensorflow.keras import Model,optimizers,layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, load_img
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,Dropout,Concatenate,Input


def dice_coef(y_true, y_pred,smooth = 10.):
    """ Dice's coefficient
    param:
        y_true: correct mask of the salamander
        y_pred: predicted mask of the salamander
    """

   # print("y_true shape {}".format(y_true.shape))
    y_true_f = K.flatten(y_true)
   # print("y_pred shape {}".format(y_pred.shape))

    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """ Dice's loss
    Args:
        y_true: correct mask of the salamander
        y_pred: predicted mask of the salamander
    """
    return 1 - dice_coef(y_true, y_pred)


def double_conv_block(prev_layer, filter_count):
    """ double_conv_block
    param:
        prev_layer: previous layer to connect to the double convolution block
        filter_count: number of filters to build in the convolution layer
    """
    new_layer = Conv2D(filter_count, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(prev_layer)
    new_layer = Conv2D(filter_count, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(new_layer)
    return new_layer

def downsample_block(prev_layer, filter_count):
    """ encoder block
    param:
        prev_layer: previous layer to connect to the double convolution block
        filter_count: number of filters to build in the convolution layer
    """
    skip_features = double_conv_block(prev_layer, filter_count)
    down_sampled = MaxPooling2D(2)(skip_features)
    #down_sampled = Dropout(0.3)(down_sampled)
    return skip_features, down_sampled

def upsample_block(prev_layer, skipped_features, n_filters):
    """ decoder block
    param:
        prev_layer: previous layer to connect to the double convolution block
        filter_count: number of filters to build in the convolution layer
    """
    upsampled = Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding="same")(prev_layer)
    upsampled = Concatenate()([upsampled, skipped_features])
    #upsampled = Dropout(0.3)(upsampled)
    upsampled = double_conv_block(upsampled, n_filters)
    return upsampled

def make_unet():

    inputs = Input(shape=(256, 256, 3))

    skipped_fmaps_1, downsample_1 = downsample_block(inputs, 64)
    skipped_fmaps_2, downsample_2 = downsample_block(downsample_1, 128)
    skipped_fmaps_3, downsample_3 = downsample_block(downsample_2, 256)
    skipped_fmaps_4, downsample_4 = downsample_block(downsample_3, 512)

    bottleneck = double_conv_block(downsample_4, 1024)

    upsample_1 = upsample_block(bottleneck, skipped_fmaps_4, 512)
    upsample_2 = upsample_block(upsample_1, skipped_fmaps_3, 256)
    upsample_3 = upsample_block(upsample_2, skipped_fmaps_2, 128)
    upsample_4 = upsample_block(upsample_3, skipped_fmaps_1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(upsample_4)
    # outputs = Conv2D(2, 1, padding="same", activation = "softmax")(upsample_4)

    unet_model = Model(inputs, outputs, name="U-Net")

    return unet_model


def dataset_paths():
    """
    returns lists of paths to images for each dataset portion i.e training and validation to correctly load dataset.

    """
    images = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/images/" + doc for doc in
              os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/images") if doc !=".DS_Store"]

    v_images = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/images/" + doc for doc in
                os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/images")if doc !=".DS_Store"]


    masks = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/masks/" + doc for doc in
             os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/masks")if doc !=".DS_Store"]

    v_masks = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/masks/" + doc for doc in
               os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/masks")if doc !=".DS_Store"]

    masks.sort()
    images.sort()
    v_masks.sort()
    v_images.sort()

    return images,masks,v_images,v_masks



def make_dataset(images,masks,v_images,v_masks,validation=False):
  """

  :param images: list of paths to training images
  :param masks: list of paths to validation images
  :param v_images: list of paths to validation images
  :param v_masks: list of paths to validation masks
  :type  validation: bool
  :return: numpy arrays of images and their corresponding masks (training set or validation set)
  """
  x = []
  y = []
  if(validation):
    for i,(image,mask) in enumerate(zip(v_images[:10000],v_masks[:10000])):
      if image==".DS_Store":
        print('found the impostor')
        pass
      print("\r"+str(i)+"/"+str(len(v_images)),end="")

      image = Image.open(os.path.join("/content/drive/MyDrive//upright_salamandre_data/val/images",image))#.convert('L')
      mask = Image.open(os.path.join("/content/drive/MyDrive//upright_salamandre_data/val/masks",mask)).convert('L')

      image = np.asarray(image.resize((256,256)))/255.
      mask = np.asarray(mask.resize((256,256)))/255.

      x.append(image)
      y.append(mask)
  else:
    for i,(image,mask) in enumerate(zip(images[:10000],masks[:10000])):
      print("\r"+str(i)+"/"+str(len(images)),end="")
      if ".DS_Store" in image:
        print('found the impostor')
        pass
      image = Image.open(os.path.join("/content/drive/MyDrive//upright_salamandre_data/train/images",image))#.convert('L')
      mask = Image.open(os.path.join("/content/drive/MyDrive//upright_salamandre_data/train/masks",mask)).convert('L')

      image = np.asarray(image.resize((256,256)))/255.
      mask = np.asarray(mask.resize((256,256)))/255.

      x.append(image)
      y.append(mask)

  return np.array(x),np.array(y)



def get_train_generator(x,y,seed=1):
    data_gen_args = dict(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow(x,batch_size=32,shuffle=True,seed=seed)
    mask_generator = mask_datagen.flow(y,batch_size=32,shuffle=True,seed=seed)
    train_generator = zip(image_generator, mask_generator)

    return train_generator

def get_val_generator(v_x,v_y,seed=1):

    image_test_datagen = ImageDataGenerator()
    mask_test_datagen = ImageDataGenerator()

    image_test_generator = image_test_datagen.flow(v_x,batch_size=32,seed=seed)

    mask_test_generator = mask_test_datagen.flow(v_y,batch_size=32,seed=seed)

    valid_generator = zip(image_test_generator, mask_test_generator)

    return valid_generator

def present_data():

    images, masks, v_images, v_masks=dataset_paths()
    register_heif_opener()
    x,y= make_dataset(images,masks,v_images,v_masks)
    v_x,v_y = make_dataset(images,masks,v_images,v_masks,True)
    y =np.expand_dims(y,axis=-1)
    v_y =np.expand_dims(v_y,axis=-1)

    return x,y,v_x,v_y

def model_info(model):
    model.summary()
    model_plotter(model)

def train_unet():

    x,y,v_x,v_y=present_data()

    plot_sample(x,y)

    u_net = make_unet()

    u_net.summary()

    train_generator= get_train_generator(x,y)
    valid_generator= get_val_generator(v_x,v_y)

    u_net.compile(tf.keras.optimizers.Adam(1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model_history = u_net.fit(train_generator, epochs=60, validation_data=valid_generator,
                              steps_per_epoch=int(x.shape[0] / 64), validation_steps=int(v_x.shape[0] / 64))

    u_net.save("u_net_run.h5")
    display_learning_curves(model_history)
    u_net.evaluate(x, y)
    u_net.evaluate(v_x, v_y)
def remove_outliers(mask, min_size=100):#from notebook "segmented_model"
    mask = (mask * 255).astype(np.uint8)

    # Find connected components in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Filter out small regions
    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered_mask[labels == label] = 255
    return filtered_mask


def dilation(msk): #from notebook "segmented_model"

    msk = (msk * 255).astype(np.uint8)

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(msk, disk)

    return img
def run_trained_model(path_model=None,path_image=None,return_result=False):
    """
        run_trained_model runs the inference of the trained model on the chosen image.
        path_model: if the parameter is provided the model is loaded from that path; if path_model is set to None the default model is loaded
        path_image:if the parameter is provided the image is loaded from that path; if path_image is set to None the default image is loaded
        return_result: if False the segmented salamander is displayed but not returned; if True the function returns the predicted mask and segmented body
    """

    # Load the saved model and specify the custom loss function

    if path_model == None:
        default_model_path="/Users/sarralaksaci/Desktop/SINF2M/TFE/u_net_upright_data-2.h5"
        with tf.keras.utils.custom_object_scope({'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}):
            loaded_model = load_model(default_model_path)
    else:
        with tf.keras.utils.custom_object_scope({'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}):
            loaded_model = load_model(path_model)
    #load the image

    if path_image==None:
        default_image_path="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/boleil_annotated/images/022.JPG"
        original = Image.open(default_image_path)
        original = np.asarray(non_stretching_resize(original, "RGB",256)) / 255.
        original = np.array(original)
        print(original.shape)
    else:
        original = Image.open(path_image)
        original = np.asarray(non_stretching_resize(original, "RGB",256)) / 255.
        original = np.array(original)


    mask = loaded_model.predict(np.expand_dims(original, axis=0))

    filtered_mask = remove_outliers(mask[0], min_size=100)
    mask = dilation(mask[0])

    segmented = np.squeeze(original).copy()
    segmented[np.squeeze(mask) < 0.3] = 0
    binary_mask = (mask > 148).astype(np.uint8)

    filtered_segmented = np.squeeze(original).copy()
    filtered_segmented[np.squeeze(filtered_mask) < 0.3] = 0


    #print(" original image shape {} \n segmented image shape {} \n mask shape {} \n binary_mask shape {}".format(
     #   original.shape, segmented.shape, mask.shape, binary_mask.shape))

    ##### Plot the results like in the notebook
    plt.imshow(filtered_segmented,cmap="gray")
    plt.show()

    fig = plt.figure(figsize=(8, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(np.squeeze(original))
    plt.title("image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(filtered_segmented, cmap="gray")
    plt.title("salamandre")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("binary mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(mask, cmap="gray")
    plt.title("mask")
    plt.axis("off")

    fig.tight_layout()
    plt.show()
    plt.close()

    if return_result:
        return binary_mask,filtered_segmented

