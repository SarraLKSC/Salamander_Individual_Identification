import os
import cv2
import random
import sklearn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Segmentation.Plots import *
from Segmentation.Preprocessing import *
from Identification.Rigid_alignment import load_batch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


val=False
BATCH=2
root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train"
v_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val"
_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated"
antiflsh_root = "/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/antiflash/"
seg = [
    "/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/antiflash/" + doc
    for doc in os.listdir(
        "/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/antiflash")]


def get_kmeans_data_bis(batch=0):
    """
     gather images to be segmented by K-means, by additional data folders boleil and stambruges
    :param batch: number of the batch to be segmented
    :return: dictionary of to_be_segmented data where the key if the file name and the values if the loaded image
   """
    start=batch*50
    end= start+50

    data_dict = {}
    if end>len(seg):
        end = len(seg)
    print(len(seg))
    for i in range(start,end):
        if ".DS_Store" not in seg[i]:
            segment = np.asarray(Image.open(os.path.join(antiflsh_root, seg[i])).convert("RGB"))
            id = seg[i].split("/")[-1]
            data_dict[id] = segment

    return data_dict

def get_kmeans_data(batch=0):
    """
    gather images to be segmented by K-means
    :param batch: number of the batch to be segmented
    :return: dictionary of to_be_segmented data where the key if the file name and the values if the loaded image
    """
    data_dict = {}

    msk_samples, _, sgmnt_samples = load_batch(batch)

    for (mask_key,mask), (sgmnt_key,sgmnt) in zip(msk_samples.items(), sgmnt_samples.items()):
        id=mask_key.split("/")[-1]
        msk, sgm = run_affine_transform(mask, sgmnt, True)
        curved, curve_angle = curved_salam(msk, True)

        if curved:
            sgm, msk = unwrap_curve(msk, sgm, True, smooth=5)

        data_dict[id]=sgm
    return data_dict

def Kmeans_post_processing(kmeans_img):
    """
    perform post-processing on color segmented image
    :param kmeans_img: spot mask produced by kmeans segmentation
    :return: corrected spot mask after morphological operations
    """
    kmeans_img=BGR_2_GRAY(kmeans_img)
    kmeans_img=opening_op(kmeans_img)

    return kmeans_img

def Kmeans(image,lab=False,eval=False):
    """
    Apply Kmeans clustering for color segmentation of an image
    :param image: image to be segmented; body of fire-salamander extracted by semantic segmentation
    :param lab: boolean defining if kmeans if applied to RGB image or its LAB converted version
    :param eval: boolean defining if evaluation of the segmentation by silhouette score is calculated
    :return: returns the binary mask for the spots and the score if eval=True
    """
    y=255
    if lab :
      image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
      y=128

    #image = cv2.medianBlur(image, 5)
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    kmeans = KMeans(n_clusters=3, n_init=10, max_iter=10).fit(reshaped)
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8), (image.shape[0], image.shape[1]))
    sorted_labels = sorted([n for n in range(2)], key=lambda x: -np.sum(clustering == x))

  # Determine which cluster corresponds to yellow spots
    cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
    yellow_spots_index = np.argmin(np.sum(np.abs(cluster_centers - [255, y, 0]), axis=1))

    # Create binary mask where yellow spots are white and everything else is black
    binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    binary_mask[clustering == yellow_spots_index] = 255

    if eval:
      sse = kmeans.inertia_
      silhouette=0
      silhouette = silhouette_score(reshaped, clustering.ravel())
      return binary_mask, sse, silhouette
    else:
      return binary_mask


def run_Kmeans(lab=False,eval=False,batch=0):
    """
    Run kmeans color segmentation on the salamander data picked up from the semantic segmentation folder; results are saved in the spot_masks folder
     :param lab: boolean defining if kmeans if applied to RGB image or its LAB converted version
     :param eval: boolean defining if evaluation of the segmentation by silhouette score is calculated
     :param batch: number of the batch to be loaded and segmented
    """
    sse_scores=[]
    silhouette_scores=[]
    sgmnt=get_kmeans_data_bis(BATCH)
    print(len(list(sgmnt.values())))
    i= 0*BATCH
    for id,salamander in sgmnt.items():
      i=i+ 1
      if eval:
        salamander_spots, sse, silhouette = Kmeans(salamander,lab,eval)
        sse_scores.append(sse)
        silhouette_scores.append(silhouette)
      else:
        salamander_spots=Kmeans(salamander,lab,eval)

      salamander_spots=Kmeans_post_processing(salamander_spots)

      # save
      if val:
          output_spot_path = v_root + "/spot_masks/" + id
          print(" output path = {}".format(output_spot_path))

          cv2.imwrite(output_spot_path, salamander_spots)
      else:
          output_spot_path = _root + "/antiflash_masks/" + id
          print(" output path = {}".format(output_spot_path))
          cv2.imwrite(output_spot_path, salamander_spots)


      plot_kmeans_seg(salamander,salamander_spots)

    print("Batch {} SSE: {}".format(batch,sum(sse_scores)/len(sgmnt)))
    print("Batch {} Silhouette: {}".format(batch,sum(silhouette_scores)/len(sgmnt)))
    return silhouette_score





