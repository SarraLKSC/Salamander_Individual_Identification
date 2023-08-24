
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tensorflow.keras.utils import plot_model
from Identification.Rigid_alignment import *
img_root = ""
msk_root = ""

def model_plotter(model):
  """
  :param model: the neural network model to plot
  """
  plot_model(
      model,
      to_file="model.png",
      show_shapes=True,
      show_dtype=True,
      show_layer_names=True,
      rankdir="TB",
      expand_nested=True,
      dpi=96,
      layer_range=None,
  )
  figure(figsize=(100,100))
  plt.imshow(np.asarray(Image.open("model.png")))
  plt.axis("off")
  plt.show()


def plot_sample(x,y):
    """
    plot a random sample from dataset composed of images and masks
    :param x: collection of images
    :param y: collection of corresponding masks
    """
    i = random.randint(0, len(x))-1
    img = x[i]
    mask = y[i]

    fig = plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(img))

    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(mask), cmap="gray")

    plt.show()


def display_learning_curves(history):
    """
    plot learning curved of trained model
    :param history: structure of training history
    """
    acc = history.history["dice_coef"]
    val_acc = history.history["val_dice_coef"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(60)

    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label="train dice_coef")
    plt.plot(epochs_range, val_acc, label="validataion dice_coef")
    plt.title("dice_coef")
    plt.xlabel("Epoch")
    plt.ylabel("dice_coef")
    plt.legend(loc="lower right")

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validataion loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.show()

def plot_predicted_seg(img_root,msk_root,images,masks):
    """
    plot the results of a random segmentation image by trained U-net
    :param img_root: root of the directory of segmented images
    :param msk_root: root of the directory of segmented masks
    :param images: collection of segmented images files
    :param masks:  collection of corresponding segmentation masks files
    :return:
    """
    i = random.randint(0, len(images) - 1)
    fig = plt.figure(figsize=(10, 7))

    img = np.asarray(Image.open(os.path.join(img_root, images[i])).convert("RGB"))
    msk = np.asarray(Image.open(os.path.join(msk_root, masks[i])).convert('L'))

    plt.subplot(1, 2, 1)
    plt.title("segmented image")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title("predicted mask")
    plt.imshow(msk, cmap="gray")
    plt.show
    nb_file = images[i].strip("val_pred")
    nb_file = nb_file.strip(".png")
    fig.subplots_adjust(hspace=0.4, top=1.1)

    fig.suptitle("sample number {}".format(nb_file))
    print(i)

def plot_colored_kmeans(image):
    plt.imshow(image)
    plt.show()

def plot_RGB_color_space(image,subsampling=1):

    """
    plot 3D RGB space for a given image
    :param image: image to express in RGB space
    :param subsampling: subsampling parameter to select pixels to display in the color space
    """
    print('Cloud point of the pixels in the RGB place (1/{} of the pixels shown):'.format(subsampling))

    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
    X = image.reshape((image.shape[0] * image.shape[1], 3))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    Xs = X[::subsampling]
    ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], marker='o', s=0.2, c=Xs)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.show()

def plot_kmeans_seg(image,spot_seg):
    """
    plot color segmentation of image
    :param image: original image
    :param spot_seg: color segmented image
    """
    fig = plt.figure(figsize=(15, 11))
    plt.subplot(1, 2, 1)
    plt.title("segmented image ")
    plt.imshow(image.astype('uint8'))

    plt.subplot(1, 2, 2)
    plt.title("segmented spots")
    plt.imshow(spot_seg.astype('uint8'), cmap="gray")
    plt.show()

def plot_kmeans_trioseg(orig,image,spot_seg):
    """
    plot color segmentation of LAB image
    :param orig: original RGB image
    :param image: LAB converted image
    :param spot_seg: results of kmeans color segmentation
    """

    fig = plt.figure(figsize=(15, 11))

    plt.subplot(1, 3, 1)
    plt.title("rgb segmented image ")
    plt.imshow(orig.astype('uint8'))

    plt.subplot(1, 3, 2)
    plt.title("lab segmented image ")
    plt.imshow(image , cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("segmented spots")
    plt.imshow(spot_seg, cmap="gray")
    plt.show()

def plot_img(img,title=""):
    """
    plot an image along with the passed title of figure
    :param img: image to display
    :param title: title to display
    """
    plt.imshow(img)
    plt.title(title)
    plt.show()
