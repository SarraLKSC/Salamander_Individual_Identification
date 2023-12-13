
#### FUNCTION CELL ####
import os
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import rotate
from pillow_heif import register_heif_opener
from skimage.measure import label, regionprops, regionprops_table

val=False
BATCH=3
DESIRED_SIZE=512

root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train"
v_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val"
_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated"

img_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/images"
msk_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/masks"

_img_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/images"
_msk_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/masks"

v_img_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/images"
v_msk_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/masks"

images =["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/images/"+ doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/images")]
v_images= ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/images/"+ doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/images")]
_images =["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/images/"+ doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/images")if ".DS_Store" not in doc]

masks =["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/masks/"+doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/masks")]
v_masks=["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/masks/"+doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/masks")]
_masks =["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/masks/"+doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/masks") if ".DS_Store" not in doc]

sgmnt_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/Boleil_ided_predictions/sgmnt"


pred_images = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/Boleil/prediction_img/"+ doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/Boleil/prediction_img")]
pred_masks = ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/Boleil/prediction_msk/"+ doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/Boleil/prediction_msk")]


pred_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/Boleil"

def initial():
    masks.sort()
    images.sort()
    v_masks.sort()
    v_images.sort()
    _masks.sort()
    _images.sort()
    pred_masks.sort()
    pred_images.sort()

def non_stretching_resize(img ,cmap ,desired_size=512):

    old_size= img.size
    ratio= float(desired_size ) /max(old_size)
    new_size =tuple([int(x * ratio) for x in old_size])
    im = img.resize(new_size, resample=Image.ANTIALIAS)

    new_im = Image.new(cmap, (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))

    img_array = np.asarray(new_im) / 255.
    return new_im

def segment(img, msk):
    """
    get segmented object from image and mask
    :param img: original image
    :param msk: segmentation mask
    :return: segmented object
    """
    image = img.copy()
    image[msk < 0.3] = 0
    return image

def load_pred(batch=0):
  initial()
  pred_msk_samples={}
  pred_sgmnt_samples={}
  start = 50 * batch
  end=start + 50

  if end>len(pred_masks):
   end=len(pred_masks)

  for i in range(batch*50,batch*50+50):

    msk_= np.asarray(Image.open(os.path.join(pred_root,pred_masks[i])).convert('L'))

    #####diff lines for ensuring right format #######
    threshold_value = 128
    msk_ = (msk_ > threshold_value).astype(int)
    #################################################
    pred_msk_samples[pred_masks[i]]=msk_
    img_= np.asarray(Image.open(os.path.join(pred_root,pred_images[i])))
    pred_sgmnt_samples[pred_images[i]]=img_

    print("\r"+str(i)+"/"+str(len(images)),end="")

  return pred_msk_samples,pred_sgmnt_samples
def load_batch(batch=0):
    initial()
    msk_samples = {}
    img_samples = {}
    sgmnt_samples = {}
    start = 50 * batch
    end=start + 50
    if end>len(_masks):
        end=len(_masks)
    #end=start+300

    if val :
        if end > len(v_masks):
            end = len(v_masks)
        for i in range(start, end):
            if ".DS_Store" in v_masks[i]: continue
            msk_ = np.asarray(Image.open(os.path.join(v_msk_root, v_masks[i])).convert('L'))
            msk_samples[v_masks[i]]=msk_
            img_ = np.asarray(Image.open(os.path.join(v_img_root, v_images[i])))
            img_samples[v_images[i]]=img_
            sgmnt_samples[v_masks[i]]=segment(img_, msk_)
            print("\r" + str(i) + "/" + str(len(v_images)), end="")
        return msk_samples, img_samples, sgmnt_samples
  #  print(len(sgmnts))
    print(start)
    print(end)
    for i in range(start, end):

        if ".DS_Store" not in _masks[i]: ##
            msk_ = np.asarray(Image.open(os.path.join(_msk_root, _masks[i])).convert('L')) ##
            msk_samples[_masks[i]] = msk_ ##

        if ".DS_Store" not in _images[i]: ##
            img_ = np.asarray(Image.open(os.path.join(_img_root, _images[i]))) ##
            img_samples[_images[i]] = img_ ##

        #print(" img path: {} ; msk path: {} ; img shape : {} ; msk shape : {}".format(_images[i],_masks[i],img_.shape,msk_.shape))
        if ".DS_Store" not in _images[i] and ".DS_Store" not in _masks[i] :
            sgmnt_samples[_masks[i]] = segment(img_, msk_)
        #if ".DS_Store" not in sgmnts[i]:
        #    sgm_= np.asarray(Image.open(os.path.join(img_root, sgmnts[i])))
        #    sgmnt_samples[sgmnts[i]]=sgm_
        print("\r" + str(i) + "/" + str(len(_images)), end="")
    return msk_samples, img_samples, sgmnt_samples

def cropping_bounds(img):
    """
      get coordinates of bounding box around the salamander mask
      img: binary mask of salamander body
      return: quadruplet of bbox coordinates
    """
    regions = regionprops(img)
    props = regions[0]
    minr, minc, maxr, maxc = props.bbox

    return minr, minc, maxc, maxr

def crop_image(img):
    """
      crop the mask based on the annotated salamander body bbox
      img: binary mask of salamander body
      return: cropped mask around the animal body (rectangle)

    """
    minr, minc, maxc, maxr = cropping_bounds(img)
    cropped = img[minr:maxr, minc:maxc]
    return cropped

def crop_image_s_mask(img, msk):
    """
      crop the image based on the annotated mask salamander body bbox
      img: image sample
      msk: corresponding binary mask
      return: cropped imaged around animal body (rectangle)

    """
    minr, minc, maxc, maxr = cropping_bounds(msk)
    cropped_img = img[minr:maxr, minc:maxc]
    return cropped_img

def zoom_bounds(img, msk, plot=True):
    """
      zoom on the image based on the annotated mask salamander body bbox
      img: image sample
      msk: corresponding binary mask
      return: cropped imaged around animal body (rectangle)
    """
    SIZE = DESIRED_SIZE # 256
    minr, minc, maxc, maxr = cropping_bounds(msk)
    newminr, newminc, newmaxc, newmaxr = minr, minc, maxc, maxr
    h = maxr - minr
    w = maxc - minc

    bound = SIZE
    if img.shape == (SIZE, SIZE):
        bound = max(h, w)
    if h <= SIZE:

        halfr = minr + h // 2
        newminr = max(halfr - bound / 2, 0)
        newmaxr = halfr + bound / 2
    else:
        bound = h
        newminr, newmaxr = minr, maxr

    if w <= h:
        halfc = minc + w // 2
        newminc = max(halfc - bound / 2, 0)
        newmaxc = min(halfc + bound / 2, img.shape[1])


    else:
        newminc, newmaxc = minc, maxc

    cropped_img = img[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]
    if plot:
        plt.imshow(cropped_img)
        plt.show()

    return newminr, newminc, newmaxc, newmaxr

def fitted_line(msk_sample,plot=True,scatter=False,ax_rotation=False):
  """
      computed fitted line that goes through salamander body mask
      msk: the binary mask of the salamander body
      plot: boolean, true if we want to splot the result
      return: the coordinates of line start and line end
  """
  contours, hierarchy = cv2.findContours(msk_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  rows, cols = msk_sample.shape

  if not contours:  #### added verification
      print("No contours found.")
      plt.imshow(msk_sample, cmap="gray")
      plt.show()
      return None, None
  contours = sorted(contours, key=cv2.contourArea, reverse=True)  ###added verification

  # Calculate endpoints of the line
  vx, vy, x, y = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)

  starty = 0

  if vy!=0:
    startx = int((starty - y) * (vx / vy) + x)
    bottomy = rows - 1
    bottomx = int((bottomy - y) * (vx / vy) + x)
  else:
    startx= bottomx =x
    bottomy = rows - 1

  # Check if the line goes out of bounds
  if startx < 0:
      startx, starty = 0, int(y - (x / vx) * vy)
  elif startx >= cols:
      startx, starty = cols - 1, int(y + ((cols - 1 - x) / vx) * vy)
  if bottomx < 0:
      bottomx, bottomy = 0, int(y - (x / vx) * vy)
  elif bottomx >= cols:
      bottomx, bottomy = cols - 1, int(y + ((cols - 1 - x) / vx) * vy)
  if plot:
  # Plot the line
    fig, ax = plt.subplots()
    ax.imshow(msk_sample, cmap='gray')
    ax.plot([startx, bottomx], [starty, bottomy], color='red', linewidth=2)
    if ax_rotation:
        midx= startx + (bottomx-startx)//2
        ax.plot( [midx,midx] , [starty,bottomy],color="blue",linewidth=2)
    if scatter:
      ax.scatter([startx],[starty],color="pink",linewidth=3)
      ax.scatter([bottomx],[bottomy],color="orange",linewidth=3)
    plt.show()
  return (startx,bottomx),(starty,bottomy)

def rotate_img(msk, angle=30, plot=True):
    # Get image size and center
    h, w = msk.shape[:2]
    cx, cy = w // 2, h // 2

    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), -int(angle), 1.0)

    # Apply rotation to image
    rotated = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_LINEAR)
    if plot:
        # Display result
        fig = plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(rotated, cmap="gray")
        plt.title("Rotated Image")
        plt.subplot(1, 2, 2)
        plt.imshow(msk, cmap="gray")
        plt.title("original image")
        plt.show()
    return rotated

def rotation_angle(msk):
    # compute angle

    (startx, bottomx), (starty, bottomy) = fitted_line(msk, plot=False)
    # Calculate the angle of the fitted line
    dy = bottomy - starty
    dx = bottomx - startx
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Calculate the angle difference between the fitted line and vertical axis
    angle_diff = angle - 90

    return -angle_diff

def apply_rotate_img(msk, replicate=False):
    alpha = rotation_angle(msk)
    if replicate:
        return alpha
    else:
        return (rotate_img(msk, alpha, False))

def replicate_rotate(img, msk):
    alpha = apply_rotate_img(msk, True)

    return (rotate_img(img, alpha, False))

def center_shift(msk):
    start, end = fitted_line(msk, False)

    # Calculate the center of the bounding box
    bbox_center = np.array([start[0], (end[1] // 2)])

    h, w = msk.shape[:2]

    # Calculate the center of the image
    mskcenter = np.array([w / 2, h / 2])

    # Calculate the shift needed to move the bounding box to the center of the image
    shift = mskcenter - bbox_center
    # Define the translation matrix
    translation_matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)

    return translation_matrix

def center_mask(msk, replicate=False):
    translation_matrix = center_shift(msk)
    h, w = msk.shape[:2]
    msk_shifted = cv2.warpAffine(msk, translation_matrix, (w, h))
    if replicate:
        return translation_matrix
    else:
        return msk_shifted

def replicate_centering(img, msk):
    translation_matrix = center_mask(msk, True)
    h, w = img.shape[:2]
    img_shifted = cv2.warpAffine(img, translation_matrix, (w, h))
    return img_shifted

def run_affine_transform(msk, img=None, replicate=False):
    """
      applied affine transformation to match desired template
      msk: binary mask to transform
      img: corresponding segmented image, used if replicate= True
      replicate: boolean, if false applies transformation to msk, if true maps msk transformation to img
      return: transformed mask (resize, rotate, scale, translate)
    """

    # resize to 256,256
    to_resize = Image.fromarray((msk * 255).astype(np.uint8))
    sample = np.array(non_stretching_resize(to_resize, "L"))

    # rotate image
    start, end = fitted_line(sample, False)
    rotated_img = apply_rotate_img(sample)

    # zoom in and crop
    newminr, newminc, newmaxc, newmaxr = zoom_bounds(rotated_img, rotated_img, False)
    cropped_img = rotated_img[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]

    # translate
    img_shifted = center_mask(cropped_img)

    if replicate:
        # print("replicate on sgmnt")
        to_resize2 = Image.fromarray((img * 1).astype(np.uint8))
        segm = np.array(non_stretching_resize(to_resize2, "RGB"))
        rotated_sgm = replicate_rotate(segm, sample)
        cropped_sgm = rotated_sgm[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]
        sgm_shifted = replicate_centering(cropped_sgm, cropped_img)

        return img_shifted, sgm_shifted
    else:
        return img_shifted

def curved_salam(msk ,return_angle=False):
    """
      returns if the salamander if curved or not
      msk: salamander mask
      return: boolean that is true
    """
    THRESHOLD =165
    h ,c , t= approx_polygone_points(msk)
    x1, y1 = h[0][0], h[0][1]
    x2, y2 = cx, cy= c
    upper_body_vec = np.array([h[0][0] - x2, h[0][1] - y2])
    lower_body_vec = np.array([t[0][0] - x2, t[0][1] - y2])
    angle = np.degrees \
        (np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
    if return_angle:
        return abs(angle ) <THRESHOLD ,angle
    else:
        return abs(angle ) <THRESHOLD

def approx_polygone_points(msk ,plot=False):

    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt =np.concatenate(contours)
    perimeter = cv2.arcLength(cnt, True)

    # Approximate the contour with a polygonal curve
    approx_curve = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    # Centroid
    M = cv2.moments(cnt)
    cx = int(M['m10' ] /M['m00'])
    cy = int(M['m01' ] /M['m00'])
    centroid = (cx ,cy)

    img = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
    approx_curve = np.append(approx_curve, [approx_curve[0]], axis=0)

    x = approx_curve[:, 0, 0]
    y = approx_curve[:, 0, 1]
    head_point =min(approx_curve ,key=lambda x: x[0][1])
    tail_point =max(approx_curve ,key=lambda x: x[0][1])
    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

        img = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
        axs[0].imshow(img)
        axs[0].scatter(head_point[0][0] ,head_point[0][1] ,color="orange" ,linewidth=2.5 ,zorder=3 ,label="head point")
        axs[0].scatter(tail_point[0][0] ,tail_point[0][1] ,color="green" ,linewidth=2.5 ,zorder=3 ,label="tail point")
        axs[0].scatter(cx ,cy ,color="pink" ,linewidth=2.5 ,zorder=3 ,label="centroid point")

        axs[0].plot(x, y, '-r', linewidth=2 ,zorder=2)
        axs[0].plot([head_point[0][0] ,cx] ,[head_point[0][1] ,cy] ,color="orange" ,linewidth=2  )  # upper body line
        axs[0].plot([tail_point[0][0] ,cx] ,[tail_point[0][1] ,cy] ,color="green" ,linewidth=2)  # lower body line
        axs[1].imshow(msk ,cmap="gray")
        upper_body_vec = np.array([head_point[0][0] - cx, head_point[0][1] - cy])
        lower_body_vec = np.array([tail_point[0][0] - cx, tail_point[0][1] - cy])
        angle = np.degrees \
            (np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
        print("Angle between upper and lower body lines: {:.2f} degrees".format(abs(angle)))

        plt.show()

    return head_point ,centroid ,tail_point

def unwrap_curve(msk ,sgm ,normalized=False ,smooth=0):

    # normalize if necessary
    if not(normalized):
        msk ,sgm =run_affine_transform(msk ,sgm ,replicate=True)

    sgm = cv2.copyMakeBorder(sgm, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value = 0)
    msk = cv2.copyMakeBorder(msk, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value = 0)

    # get required points
    head_point ,centroid ,tail_point = approx_polygone_points(msk)
    x1, y1 = head_point[0][0], head_point[0][1]
    x2, y2 = cx, cy = centroid[0] ,centroid[1]
    #print("INSIDE UNWARP: centroid={} ; head_point= {} ; tail_point= {} ".format(centroid ,head_point ,tail_point))
    # computer angle between upper body vect and vertical axis
    angle = np.arctan2(x2 - x1, y2 - y1) * 180 / np.pi

    # split segment int upper and lower based on centroid
    upper_bod =sgm[:cy ,:]
    lower_bod =sgm[cy: ,:]
    upper_bod_m =msk[:cy ,:]
    lower_bod_m =msk[cy: ,:]
    # rotate uppder bod
    rotated_upper_bod = rotate(upper_bod ,-angle)
    rotated_upper_bod =(rotated_upper_bod *255).astype(np.uint8)
    rotated_upper_bod_m = rotate(upper_bod_m ,-angle)
    rotated_upper_bod_m =(rotated_upper_bod_m *255).astype(np.uint8)

    # trnaslate lower bod

    shiftx =cx- head_point[0][0]
    if shiftx >0:
        #print("to the left")
        shiftx =shiftx -smooth
    else:
        shiftx =shiftx +smooth
        #print("to the right")

    M = np.float32([[1 ,0 ,-(shiftx)],
                    [0 ,1 ,0]])
    rows ,cols ,ch = lower_bod.shape
    # print(shiftx)

    lower_bod =cv2.warpAffine(lower_bod ,M ,(cols ,rows))
    lower_bod_m =cv2.warpAffine(lower_bod_m ,M ,(cols ,rows))

    # concatenate upper and lower
    new_salam_sgm =np.concatenate((rotated_upper_bod ,lower_bod) ,axis=0)
    new_salam_msk =np.concatenate((rotated_upper_bod_m ,lower_bod_m) ,axis=0)

    return(new_salam_sgm ,new_salam_msk)

def sheer(msk,sgm,normalized=False,normalize=True):

  # transform if necessary
  if not(normalized):
    msk,sgm=run_affine_transform(msk,sgm,replicate=True)

  # expand black background in case the foreground moves off the borders
  sgm = cv2.copyMakeBorder(sgm, 120, 10, 120, 0, cv2.BORDER_CONSTANT, None, value = 0)
  msk = cv2.copyMakeBorder(msk, 120, 10, 120, 0, cv2.BORDER_CONSTANT, None, value = 0)
  # sheer transform
  M = np.float32([[1,-0.7,0],
                [0,1,0]])
  rows,cols,ch = sgm.shape
  sheer_sgm = cv2.warpAffine(sgm, M,(cols,rows))
  sheer_msk = cv2.warpAffine(msk, M,(cols,rows))

  if normalize:
    sheer_msk,sheer_sgm=run_affine_transform(sheer_msk,sheer_sgm,True)

  return sheer_msk,sheer_sgm

def rigid_registration_test():
    print("rigid registration test")
    initial()
    ###msk_samples, img_samples, sgmnt_samples= load_batch(BATCH)

    msk_samples,sgmnt_samples= load_pred(BATCH) ## loaded predictions with correct format enforced
    plot_affine_transform_sgmnt(msk_samples,sgmnt_samples)
    #plot_affine_transform_msk(msk_samples,sgmnt_samples)
    #plot_affine_transform_msk_sgmnt(msk_samples,sgmnt_samples)

    #### curved salamander opretations####
    #plot_unwrap_curve(msk_samples,sgmnt_samples)
    # plot_OpenCV_contour_results(msk_samples)
    # plot_curve_detect_approx_polygone(msk_samples,sgmnt_samples)
    # plot_msk_approx_polygone(msk_samples)

################################### - PLOTS - #######################################
################################### - PLOTS - #######################################
################################### - PLOTS - #######################################
################################### - PLOTS - #######################################


def plot_affine_transform_sgmnt(msk_samples, sgmnt_samples):
    # run transformations on segments : rotate and scale and translation

    for sample, segm in zip(msk_samples.values(), sgmnt_samples.values()):
        # resize to 256,256
        to_resize = Image.fromarray((sample * 255).astype(np.uint8))
        sample = np.array(non_stretching_resize(to_resize, "L"))
        to_resize2 = Image.fromarray((segm * 1).astype(np.uint8))
        segm = np.array(non_stretching_resize(to_resize2, "RGB"))

        # rotate image
        start, end = fitted_line(sample, False)
        rotated_img = apply_rotate_img(sample)
        rotated_sgm = replicate_rotate(segm, sample)

        # zoom in and crop
        newminr, newminc, newmaxc, newmaxr = zoom_bounds(rotated_img, rotated_img, False)
        cropped_img = rotated_img[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]
        cropped_sgm = rotated_sgm[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]

        # translate
        img_shifted = center_mask(cropped_img)
        sgm_shifted = replicate_centering(cropped_sgm, cropped_img)

        fig = plt.figure(figsize=(25, 20))
        plt.subplot(1, 5, 1)
        plt.imshow(segm)
        plt.title('original segmnt')
        plt.axis("off")

        plt.subplot(1, 5, 2)
        plt.imshow(segm)
        start, end = fitted_line(sample, False)
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.title("original segmnt w.fitted line")
        plt.axis("off")

        plt.subplot(1, 5, 3)
        plt.imshow(rotated_sgm)
        plt.title("rotated segmnt")
        plt.axis("off")

        plt.subplot(1, 5, 4)
        plt.imshow(cropped_sgm, cmap="gray")
        plt.title("cropped segmnt")
        plt.axis("off")

        plt.subplot(1, 5, 5)

        plt.imshow(sgm_shifted, cmap="gray")
        plt.title("centered segmnt")
        plt.axis("off")
        plt.show()
        plt.show()
        plt.tight_layout()


def plot_affine_transform_msk_sgmnt(msk_samples, sgmnt_samples):
    # run transformations on samples of mask and segment : rotate and scale and translation
    for (sample_key,sample), (segm_key,segm) in zip(msk_samples.items(), sgmnt_samples.items()):

        segm_key_id= segm_key.split("/")[-1]
        sample_key_id= sample_key.split("/")[-1]
        # resize to 256,256
        to_resize = Image.fromarray((sample * 255).astype(np.uint8))
        sample = np.array(non_stretching_resize(to_resize, "L"))

        ######### diff lines to ensure right format ############
        threshold_value = 128
        sample = (sample > threshold_value).astype(np.uint8)

        if np.sum(sample) == 0:
            continue
        #######################################################

        to_resize2 = Image.fromarray((segm * 1).astype(np.uint8))
        segm = np.array(non_stretching_resize(to_resize2, "RGB"))

        # rotate image
        start, end = fitted_line(sample, False)
        rotated_img = apply_rotate_img(sample)
        rotated_sgm = replicate_rotate(segm, sample)

        # zoom in and crop
        newminr, newminc, newmaxc, newmaxr = zoom_bounds(rotated_img, rotated_img, False)
        cropped_img = rotated_img[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]
        cropped_sgm = rotated_sgm[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]

        # translate
        img_shifted = center_mask(cropped_img)
        sgm_shifted = replicate_centering(cropped_sgm, cropped_img)

        #save
        if val:
            img_shifted_scaled = img_shifted * 255
            img_shifted_uint8 = img_shifted_scaled.astype(np.uint8)
            output_spot_path = v_root + "/msk/" + sample_key_id
            cv2.imwrite(output_spot_path, img_shifted_uint8)
            output_sgmnt_path = v_root + "/sgmnt/" + sample_key_id
            print(output_spot_path)
            cv2.imwrite(output_sgmnt_path, cv2.cvtColor(sgm_shifted, cv2.COLOR_BGR2RGB))

        else:
            img_shifted_scaled = img_shifted * 255
            img_shifted_uint8 = img_shifted_scaled.astype(np.uint8)

            output_spot_path =_root+ "/msk/"+sample_key_id
            cv2.imwrite(output_spot_path, img_shifted_uint8)
            output_sgmnt_path =_root+ "/sgmnt/"+segm_key_id
            print(output_spot_path)
            cv2.imwrite(output_sgmnt_path,  cv2.cvtColor(sgm_shifted, cv2.COLOR_BGR2RGB))

        fig = plt.figure(figsize=(20, 20))
        plt.subplot(1, 5, 1)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(sample)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.imshow(sample, cmap="gray")
        plt.title('original mask: {}'.format(sample_key_id))
        plt.axis("off")

        plt.subplot(1, 5, 2)
        plt.imshow(sample, cmap="gray")
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.title("original mask w.fitted line")
        plt.axis("off")

        plt.subplot(1, 5, 3)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(rotated_img)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.imshow(rotated_img, cmap="gray")
        plt.title("rotated mask")
        plt.axis("off")

        plt.subplot(1, 5, 4)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(cropped_img)
        start, end = fitted_line(cropped_img, False)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.imshow(cropped_img, cmap="gray")
        plt.title("cropped mask")
        plt.axis("off")

        plt.subplot(1, 5, 5)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(img_shifted)
        start, end = fitted_line(img_shifted, False)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2)
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.imshow(img_shifted, cmap="gray")
        plt.title("centered mask")
        plt.axis("off")
        plt.show()
        plt.show()

        #### SAVE TRANSFORMED msk
        plt.tight_layout()

        fig = plt.figure(figsize=(25, 20))
        plt.subplot(1, 5, 1)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(sample)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.imshow(segm)
        plt.title('original segmnt: {}'.format(segm_key_id))
        plt.axis("off")

        plt.subplot(1, 5, 2)
        plt.imshow(segm)
        start, end = fitted_line(sample, False)
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.title("original segmnt w.fitted line")
        plt.axis("off")

        plt.subplot(1, 5, 3)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(rotated_img)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.imshow(rotated_sgm)
        plt.title("rotated segmnt")
        plt.axis("off")

        plt.subplot(1, 5, 4)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(cropped_img)
        start, end = fitted_line(cropped_img, False)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.imshow(cropped_sgm, cmap="gray")
        plt.title("cropped segmnt")
        plt.axis("off")

        plt.subplot(1, 5, 5)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(img_shifted)
        start, end = fitted_line(img_shifted, False)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2)
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.imshow(sgm_shifted, cmap="gray")
        plt.title("centered segmnt")
        plt.axis("off")
        plt.show()
        plt.show()
        #### SAVE TRANSFORMED sgmnt

        plt.tight_layout()

def plot_affine_transform_msk(msk_samples,sgmnt_samples):
    # run transformations on samples : rotate and scale and translation

    for sample in msk_samples.values():
        # resize to 256,256
        to_resize = Image.fromarray((sample * 255).astype(np.uint8))
        sample = np.array(non_stretching_resize(to_resize, "L"))

        ######### diff lines to ensure right format ############
        threshold_value = 128
        sample = (sample > threshold_value).astype(np.uint8)
        if np.sum(sample) == 0:
            continue
        #######################################################
        # rotate image
        start, end = fitted_line(sample, False)
        rotated_img = apply_rotate_img(sample)

        # zoom in and crop
        newminr, newminc, newmaxc, newmaxr = zoom_bounds(rotated_img, rotated_img, False)
        cropped_img = rotated_img[int(newminr):int(newmaxr), int(newminc):int(newmaxc)]

        # translate
        img_shifted = center_mask(cropped_img)

        fig = plt.figure(figsize=(15, 10))
        plt.subplot(1, 5, 1)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(sample)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.imshow(sample, cmap="gray")
        plt.title('original mask')
        plt.axis("off")

        plt.subplot(1, 5, 2)
        plt.imshow(sample, cmap="gray")
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.title("original mask w.fitted line")
        plt.axis("off")

        plt.subplot(1, 5, 3)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(rotated_img)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.imshow(rotated_img, cmap="gray")
        plt.title("rotated mask")
        plt.axis("off")

        plt.subplot(1, 5, 4)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(cropped_img)
        start, end = fitted_line(cropped_img, False)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.imshow(cropped_img, cmap="gray")
        plt.title("cropped mask")
        plt.axis("off")

        plt.subplot(1, 5, 5)
        newminr, newminc, newmaxc, newmaxr = cropping_bounds(img_shifted)
        start, end = fitted_line(img_shifted, False)

        bx = (newminc, newmaxc, newmaxc, newminc, newminc)
        by = (newminr, newminr, newmaxr, newmaxr, newminr)
        plt.plot(bx, by, '-b', linewidth=2)
        plt.plot(list(start), list(end), color="red", linewidth=2)
        plt.imshow(img_shifted, cmap="gray")
        plt.title("centered mask")
        plt.axis("off")

        plt.show()
        plt.tight_layout()


def plot_unwrap_curve(msk_samples,sgmnt_samples):
    i = 0
    cpt = 0
    for mask, sgmnt in zip(msk_samples.values(), sgmnt_samples.values()):
        i += 1
        salam_msk, salam_sgm = run_affine_transform(mask, sgmnt, True)
        curved, curve_angle = curved_salam(salam_msk, True)
        if curved:
            cpt += 1
            print("sample {} ".format(i - 1))
            head_point, centroid, tail_point = approx_polygone_points(salam_msk)
            x1, y1 = head_point[0][0], head_point[0][1]
            x2, y2 = cx, cy = centroid
            angle = np.arctan2(x2 - x1, y2 - y1) * 180 / np.pi

            plt.figure(figsize=(15, 15))
            plt.subplot(1, 5, 1)
            plt.imshow(salam_sgm)
            plt.axis("off")

            plt.subplot(1, 5, 2)
            plt.imshow(salam_sgm)
            plt.plot([head_point[0][0], cx], [head_point[0][1], cy], color="orange", linewidth=2)  # upper body line
            plt.plot([tail_point[0][0], cx], [tail_point[0][1], cy], color="green", linewidth=2)  # lower body line
            plt.scatter(head_point[0][0], head_point[0][1], color="orange", linewidth=2.5, zorder=3, label="head point")
            plt.scatter(tail_point[0][0], tail_point[0][1], color="green", linewidth=2.5, zorder=3, label="tail point")
            plt.scatter(cx, cy, color="pink", linewidth=2.5, zorder=3, label="centroid point")
            plt.axis("off")
            plt.axvline(x=salam_sgm.shape[1] // 2, color='red', linestyle='--')
            plt.axhline(y=cy, color="pink", linestyle="--")

            sgm, msk = unwrap_curve(salam_msk, salam_sgm, True, smooth=5)
            head_point, centroid, tail_point = approx_polygone_points(msk)
            cx, cy = centroid

            plt.subplot(1, 5, 3)
            plt.imshow(sgm)
            plt.axis("off")

            plt.subplot(1, 5, 4)
            plt.imshow(msk, cmap="gray")
            plt.axis("off")
            _, curve_angle2 = curved_salam(msk, True)
            print("old angle {:.2f} ; new angle {:.2f}".format(abs(curve_angle), abs(curve_angle2)))

            plt.subplot(1, 5, 5)
            plt.imshow(sgm)
            plt.plot([head_point[0][0], cx], [head_point[0][1], cy], color="orange", linewidth=2)  # upper body line
            plt.plot([tail_point[0][0], cx], [tail_point[0][1], cy], color="green", linewidth=2)  # lower body line
            plt.scatter(head_point[0][0], head_point[0][1], color="orange", linewidth=2.5, zorder=3, label="head point")
            plt.scatter(tail_point[0][0], tail_point[0][1], color="green", linewidth=2.5, zorder=3, label="tail point")
            plt.scatter(cx, cy, color="pink", linewidth=2.5, zorder=3, label="centroid point")
            plt.axis("off")

            plt.show()

    print("totale of {} curved salamanders in batch of {} ".format(cpt, i - 1))


def plot_OpenCV_contour_results(msk_samples):
    # plot openCV contour options on mask samples

    for mask in msk_samples.values():
        mask = run_affine_transform(mask)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute convex hull of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)
        # print(hull.shape)

        # plot subplots
        fig, axs = plt.subplots(ncols=4, figsize=(10, 5))

        # plot mask
        axs[0].imshow(mask, cmap='gray')
        axs[0].set_title("Mask")

        # plot defects
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Convexity Defects")
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(largest_contour[s][0])
            end = tuple(largest_contour[e][0])
            far = tuple(largest_contour[f][0])
            axs[1].plot([start[0], end[0]], [start[1], end[1]], color='green', linewidth=2)
            axs[1].plot(far[0], far[1], 'ro')
        # plot convex hull
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute convex hull of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)

        axs[2].imshow(mask, cmap='gray')
        axs[2].set_title("Convex Hull")
        axs[2].plot(largest_contour[:, 0, 0], largest_contour[:, 0, 1], '-r', linewidth=2.5)
        axs[2].plot(hull[:, 0, 0], hull[:, 0, 1], '-b', linewidth=2.5)

        # Compute Poly approx
        cnt = np.concatenate(contours)
        perimeter = cv2.arcLength(cnt, True)

        # Approximate the contour with a polygonal curve
        approx_curve = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        # Centroid
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
        approx_curve = np.append(approx_curve, [approx_curve[0]], axis=0)

        # Extract the x and y coordinates of the approximated curve
        x = approx_curve[:, 0, 0]
        y = approx_curve[:, 0, 1]
        head_point = min(approx_curve, key=lambda x: x[0][1])
        tail_point = max(approx_curve, key=lambda x: x[0][1])

        # Plot the original image and the approximated curve
        axs[3].imshow(img)
        axs[3].scatter(head_point[0][0], head_point[0][1], color="orange", linewidth=2.5, zorder=3, label="head point")
        axs[3].scatter(tail_point[0][0], tail_point[0][1], color="green", linewidth=2.5, zorder=3, label="tail point")
        axs[3].scatter(cx, cy, color="pink", linewidth=2.5, zorder=3, label="centroid point")

        axs[3].plot(x, y, '-r', linewidth=2, zorder=2)
        axs[3].plot([head_point[0][0], cx], [head_point[0][1], cy], color="orange", linewidth=2)  # upper body line
        axs[3].plot([tail_point[0][0], cx], [tail_point[0][1], cy], color="green", linewidth=2)  # lower body line
        # Compute and print the angle between upper and lower body lines
        upper_body_vec = np.array([head_point[0][0] - cx, head_point[0][1] - cy])
        lower_body_vec = np.array([tail_point[0][0] - cx, tail_point[0][1] - cy])
        angle = np.degrees(
            np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
        print("Angle between upper and lower body lines: {:.2f} degrees".format(abs(angle)))

        # axs[3].legend()
        axs[3].set_title("Approximated Curve")

        plt.show()

def plot_curve_detect_approx_polygone(msk_samples, sgmnt_samples):
    # plot approx polygone for smg samples
    i = 0
    cpt = 0

    for mask, sgm in zip(msk_samples.values(), sgmnt_samples.values()):
        i += 1
        mask, sgm = run_affine_transform(mask, sgm, True)

        # plot subplots
        fig, axs = plt.subplots(ncols=3, figsize=(10, 5))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute Poly approx
        cnt = np.concatenate(contours)
        perimeter = cv2.arcLength(cnt, True)

        # Approximate the contour with a polygonal curve
        approx_curve = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        # Centroid
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
        approx_curve = np.append(approx_curve, [approx_curve[0]], axis=0)

        # Extract the x and y coordinates of the approximated curve
        x = approx_curve[:, 0, 0]
        y = approx_curve[:, 0, 1]
        head_point = min(approx_curve, key=lambda x: x[0][1])
        tail_point = max(approx_curve, key=lambda x: x[0][1])

        # Plot the original image and the approximated curve
        axs[0].imshow(img)
        axs[0].scatter(head_point[0][0], head_point[0][1], color="orange", linewidth=2.5, zorder=3, label="head point")
        axs[0].scatter(tail_point[0][0], tail_point[0][1], color="green", linewidth=2.5, zorder=3, label="tail point")
        axs[0].scatter(cx, cy, color="pink", linewidth=2.5, zorder=3, label="centroid point")

        axs[0].plot(x, y, '-r', linewidth=2, zorder=2)
        axs[0].plot([head_point[0][0], cx], [head_point[0][1], cy], color="orange", linewidth=2)  # upper body line
        axs[0].plot([tail_point[0][0], cx], [tail_point[0][1], cy], color="green", linewidth=2)  # lower body line
        # Compute and print the angle between upper and lower body lines
        upper_body_vec = np.array([head_point[0][0] - cx, head_point[0][1] - cy])
        lower_body_vec = np.array([tail_point[0][0] - cx, tail_point[0][1] - cy])
        angle = np.degrees(
            np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
        print("Angle between upper and lower body lines: {:.2f} degrees".format(abs(angle)))
        print(" sample n:{} ".format(i))
        # axs[3].legend()
        axs[0].set_title("Approximated msk Curve")
        axs[0].axis("off")

        axs[1].imshow(sgm)
        axs[1].plot([head_point[0][0], cx], [head_point[0][1], cy], color="orange", linewidth=2)  # upper body line
        axs[1].plot([tail_point[0][0], cx], [tail_point[0][1], cy], color="green", linewidth=2)  # lower body line
        axs[1].scatter(head_point[0][0], head_point[0][1], color="orange", linewidth=2.5, zorder=3, label="head point")
        axs[1].scatter(tail_point[0][0], tail_point[0][1], color="green", linewidth=2.5, zorder=3, label="tail point")
        axs[1].scatter(cx, cy, color="pink", linewidth=2.5, zorder=3, label="centroid point")
        axs[1].set_title("Approximated img Curve")
        axs[1].axis("off")

        axs[2].imshow(sgm)
        if abs(angle) > 165:
            axs[2].axis("off")
            axs[2].set_title("straight salamander", color="green")

        else:
            axs[2].axis("off")
            axs[2].set_title("curved salamander", color="red")
            cpt += 1

        plt.show()
    print("totale of {} curved salamanders in batch of {} ".format(cpt, i - 1))

def plot_msk_approx_polygone(msk_samples,i=5):
    # plot approx polygone for a given salam mask

    # Load the mask
    msk = run_affine_transform(msk_samples.values()[i])

    # Find contours
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.concatenate(contours)
    # Loop through the contours and detect curves
    if True:
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(cnt, True)

        # Approximate the contour with a polygonal curve
        approx_curve = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        print("shape approx_curve {}".format(approx_curve.shape))
        # Centroid
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Check if the number of vertices in the approximated curve is less than the number of vertices in the contour
        # Draw the contour on a copy of the original image
        img = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
        approx_curve = np.append(approx_curve, [approx_curve[0]], axis=0)

        # Extract the x and y coordinates of the approximated curve
        x = approx_curve[:, 0, 0]
        y = approx_curve[:, 0, 1]
        head_point = min(approx_curve, key=lambda x: x[0][1])
        tail_point = max(approx_curve, key=lambda x: x[0][1])

        # Plot the original image and the approximated curve
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.scatter(head_point[0][0], head_point[0][1], color="orange", linewidth=2.5, zorder=3, label="head point")
        ax.scatter(tail_point[0][0], tail_point[0][1], color="green", linewidth=2.5, zorder=3, label="tail point")
        ax.scatter(cx, cy, color="pink", linewidth=2.5, zorder=3, label="centroid point")

        ax.plot(x, y, '-r', linewidth=2, zorder=2)
        ax.plot([head_point[0][0], cx], [head_point[0][1], cy], color="orange", linewidth=2)  # upper body line
        ax.plot([tail_point[0][0], cx], [tail_point[0][1], cy], color="green", linewidth=2)  # lower body line
        # Compute and print the angle between upper and lower body lines
        upper_body_vec = np.array([head_point[0][0] - cx, head_point[0][1] - cy])
        lower_body_vec = np.array([tail_point[0][0] - cx, tail_point[0][1] - cy])
        angle = np.degrees(
            np.arctan2(np.linalg.det([upper_body_vec, lower_body_vec]), np.dot(upper_body_vec, lower_body_vec)))
        print("Angle between upper and lower body lines: {:.2f} degrees".format(abs(angle)))

        ax.legend()
        ax.set_title(" Image' Approximated Curve")
        plt.show()

