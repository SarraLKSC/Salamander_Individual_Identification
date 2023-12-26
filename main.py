import os
import psutil
import matplotlib.pyplot as plt
from Segmentation.Unet import *
from Segmentation.Kmeans import *
from Segmentation.Plots import *
from Segmentation.Preprocessing import *
from Identification.LBP import *
from Identification.identity_ETL import *
from Segmentation.Preprocessing import *
import tensorflow as tf
from Identification.Rigid_alignment import *
#if __name__ == '__main__':
# Get the system's memory information


segmnts_root = "/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/sgmnt/"
antiflsh_root = "/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/antiflash/"

sgmnts = [
    "/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/sgmnt/" + doc
    for doc in os.listdir(
        "/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/stambruges_annotated/sgmnt")]

memory = psutil.virtual_memory()

# Print the available memory in bytes
print("Available memory:", memory.available)
# Print the available memory in human-readable format
print("Available memory:", psutil._common.bytes2human(memory.available))

## different modules of the project can be tested here by uncommenting the corresponding line of code.
## data must be availalbe for the code to work as each step requires loading a batch of images to run on


#rigid_registration_test()
#print(tf.__version__)
## get mask and segmeted body from run_trained_model; set return_result to True for that
msk,sgmnt=run_trained_model(return_result=True)

## get transformed mask and segmented body from affine_transform_msk_sgmnt
msk,sgmnt= affine_transform_msk_sgmnt(msk,sgmnt)

plt.imshow(msk,cmap="gray")
plt.show()
plt.imshow(sgmnt)
plt.show()


#run_Kmeans(True,False,0)

#plot_LBP_encoding()

#plot_LBP_hist()


#plot_LBP_encoding_radius_variation(8,[1,2,3,4])

#plot_LBP_encoding_points_variation([5,15,25,30],3)

#plot_one2many()

#plot_one2many_top5()


#run_antiflash(segmnts_root,antiflsh_root,sgmnts)





