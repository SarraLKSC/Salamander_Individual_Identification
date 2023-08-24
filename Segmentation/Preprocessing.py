import matplotlib.pyplot as plt
import numpy as np
from heic2png import HEIC2PNG
from Segmentation.Plots import *
import random
from Segmentation.Kmeans import *

def conv_heic2png(root):
    """
    converts all images present at root from heic format to png format
    :param root: root of the directory which contains the heic images
    :return:None
    """
    files=os.listdir(root)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    for f in files:
        heic_img = HEIC2PNG(os.path.join(root,f))
        heic_img.save()


def conv_png_jpg(root):
    """
     converts all images present at root from png format to jpg format
    :param root: root of the directory which contains the png images
    :return:None
    """
    files=os.listdir(root)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    for f in files:
        png_img = Image.open(os.path.join(root,f))
        png_img=png_img.convert("RGB")
       # print(type(png_img))
        new_f=f.strip(".png")
        new_f=new_f+".jpg"
        #print("f : "+f+" new f : "+new_f)
        png_img.save(os.path.join(root,new_f))


def closing_op(msk,iter=1):
    """
    performs morphological closing on the image
    :param msk: binary image to process
    :param iter: number of iterations of closing
    :return: closed msk
    """
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
    img_close = cv2.morphologyEx(msk,cv2.MORPH_CLOSE, kernel,iterations=iter)

    return img_close

def opening_op(msk,iter=1):
    """
     performs morphological opening on the image
    :param msk: binary image to process
    :param iter: number of iterations of opening
    :return: opened msk
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    img_open = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel, iterations=iter)

    return img_open

def BGR_2_GRAY(img):
    """
    convert BGR image to GRAY
    :param img: image to convert
    :return: converted gray image
    """
    if img.ndim == 2: # if already in grayscale
        return img
    img = np.float32(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def RGB_2_HSV(img):
    """
    convert RGB image to HSC
    :param img: image to convert
    :return: converted HSV image
    """
    #print(type(img))
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    return hsv

def BGR_2_LAB(img):
    """
    convert BGR image to LAB
    :param img: image to convert
    :return: converted LAB image
    """
    img=np.float32(img)
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    return lab

def LAB_RGB(img):
    """
    convert lab image to rgb
    :param img: image to convert
    :return: convertes rgb image
    """
    img=np.float32(img)
    rgb=cv2.cvtColor(img,cv2.COLOR_LAB2RGB)
    return rgb

def valid_path(path):
    """
    verify if the path is a valid aka not the ".DS_Store" file of macOS directories"
    :param path:
    :return: boolean value
    """
    return (".DS_Store" not in path )

def load_lab_image(path):
    """
    load image from path in LAB color space
    :param path: path to the image
    :return: image in LAB format
    """
    rgb_img=cv2.imread(path)
    rgb_img=np.array(rgb_img)/255
    rgb_img=np.float32(rgb_img)
    lab_img=cv2.cvtColor(rgb_img,cv2.COLOR_BGR2LAB)

    return lab_img

def load_rgb_image(path):
    """
    load image from path in RGB color space
    :param path: path to the image
    :return: image in RGB format
    """
    img = cv2.imread(path)
    img = np.array(img) / 255
    img=np.float32(img)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return rgb_img
def load_gray_image(path):
    """
       load image from path in gray color space
       :param path: path to the image
       :return: image in gray scale format
       """
    img= cv2.imread(path)
    img= np.array(img)/255
    img=np.float32(img)
    gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray_img

#https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
def load_n_resize(path,stretch=True,desired_size=256):

    if stretch:
        img=Image.open(path).convert("RGB")
        old_size = img.size  # old_size[0] is in (width, height) format
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = img.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

        img_array=np.asarray(new_im)/255.
        plt.imshow(img_array)
        plt.title("non destructive")
        plt.show()
        return(img_array)
    else :
        img=Image.open(path).convert("RGB")
        img=img.resize((256,256))
        plt.imshow(img)
        plt.title("non destructive")
        plt.show()
        return (img)


def non_stretching_resize(img,desired_size=256):
    """
    resive image in a non-destructive way
    :param img: image to resize
    :param desired_size: dimensions of new image
    :return: resized image
    """
    old_size= img.size
    ratio= float(desired_size)/max(old_size)
    new_size =tuple([int(x * ratio) for x in old_size])

    im = img.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

    img_array=np.asarray(new_im)/255.
    return new_im


def anti_flash(input_path):
    """
    Glare correction on segmented salamanders
    :param input_path: path of the image to correct
    :return: corrected image
    """
    image_in = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV))
    #plt.imshow(image_in)
    nonSat = s < 180
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
    v2 = v.copy()
    v2[nonSat == 0] = 0
    glare = v2 > 150
    glare = cv2.dilate(glare.astype(np.uint8), disk)
    glare = cv2.dilate(glare.astype(np.uint8), disk)

    corrected = cv2.inpaint(image_in, glare, 5, cv2.INPAINT_NS)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_in)
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.imshow(glare, cmap="gray")
    plt.title("Glare Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(corrected)
    id=input_path.split("/")[-1]
    plt.title("Corrected: {}".format(id))
    plt.show()
    return corrected

# Example usage
def run_antiflash(segmnts_root,antiflsh_root,sgmnts):
    """
    Run antiflash correction on all salamander segmented bodies present in inputted directory
    :param segmnts_root: directory of images to correct
    :param antiflsh_root: target directory after correction
    :param sgmnts: list of files of correct

    """
    random.shuffle(sgmnts)
    for i, seg in enumerate(sgmnts):
        if ".DS_Store" not in seg:
            segment = np.asarray(Image.open(os.path.join(segmnts_root, seg)).convert("RGB"))
            antiflash_segment = anti_flash(seg)
            output_path= os.path.join(antiflsh_root,seg.replace("sgmnt","antiflash"))
            print("outputpath = {}".format(output_path))
            cv2.imwrite(output_path,  cv2.cvtColor(antiflash_segment, cv2.COLOR_BGR2RGB))

    print("end total "+str(i))