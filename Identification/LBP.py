from Segmentation.Unet import *
from Segmentation.Kmeans import *
from Segmentation.Plots import *
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.transform import resize
import numpy as np

from skimage import data
from skimage.color import label2rgb

# settings for LBP
radius = 1
n_points = 8 * radius

BATCH=0
val=False
root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train"
v_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val"

spot_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/spot_masks"
v_spot_root="/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/spot_masks"

spots =["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/spot_masks/"+ doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/spot_masks")]
v_spots= ["/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/train/spot_masks/"+ doc for doc in os.listdir("/Users/sarralaksaci/Desktop/SINF2M/TFE/data/updated_upright_salamandre_data/val/spot_masks")]



def block_bounds(img_size=256, block_size=8):
    """
      returns the bounds of the blocks to split image of img_size into block_size
    """
    bounds_dict = {}
    bounds_dict[0] = init = (0, 0)
    nb_blocks = int(img_size / block_size)
    for b in range(1, nb_blocks ** 2):
        i, j = init
        if j + block_size >= img_size and i < img_size:
            j = 0
            i = i + block_size
        else:
            if i < img_size:
                j = j + block_size
            else:
                break
        bounds_dict[b] = init = (i, j)
    return bounds_dict


def get_lbp(image, radius=2, n_points=2* 8):
    """
      returns Local Binary Pattern encoded image with passed parameters
    """
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp



def get_cslbp(image, radius=3, n_points=2 * 8):
    rows, cols = image.shape # get shape of image
    cslbp_image = np.zeros_like(image, dtype=np.uint16) # create placeholder matrix for the LBP encoded image

    angles = 2 * np.pi * np.arange(n_points) / n_points # calcules the angles of samples around the circle

    for i in range(radius, rows - radius): #excluding points over radius
        for j in range(radius, cols - radius): #excluding points over radius
            neighbors = []
            for k in range(n_points): #collecting n_points neighbors
                #get indices of sample image based on the angle calculated
                x = int(np.round(i + radius * np.cos(angles[k])))
                y = int(np.round(j - radius * np.sin(angles[k])))

                neighbors.append(image[x, y]) # stpre neighbor pixel

            cslbp = np.uint32(0)  # Cast to np.uint32 to avoid overflow
            for k in range(n_points - 1):
                cslbp += s(neighbors[k] - neighbors[k + 1])
                #cslbp value calculated as the sum of the pairs of neighbors where neigh[k]>= neigh[k+1]

            cslbp_image[i, j] = cslbp

    return cslbp_image


def s(x):
    return 1 if x >= 0 else 0


def get_hist_LBP(lbp):
    """
      return histogram of lbp encoded image
    """
    n_bins = 10  # lbp.max() + 1
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=int(n_bins), range=(0, n_bins))
    return hist


def blocks(lbp_image, idx=None):
    """
      return list of all the local histograms corresponding to each block of the passed lbp encoded image
    """
    bounds = block_bounds()

    if idx != None:
        i, j = bounds[idx]
        tmp = lbp_image.copy()
        tmp[i:i + 32, j:j + 32] = 0
        plt.imshow(tmp)
        plt.show()
    else:
        hists = []
        for pair in bounds.values():
            i, j = pair
            block = lbp_image[i:i + 32, j:j + 32]
            hists.append(get_hist_LBP(block))
        return hists


def feature_hist(hists):
    """
      return concatenated histogram of passed list of block histograms
    """
    feature_vector = []
    offset = 0
    for i, h in enumerate(hists):
        feature_vector.append(h)
    return np.concatenate(feature_vector)

# Function to calculate Chi-distance
def chi2_distance(A, B):
    """
    Return the chi2 distance of distributions A and B.
    """
    epsilon = 1e-10  # Small epsilon value to avoid division by zero
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b + epsilon) for (a, b) in zip(A, B)])

    return chi


def kullback_leibler_divergence(p, q):
    """
      return KL distance of distributions p and q
    """
    p = np.asarray(p)
    q = np.asarray(q)
    if (p.shape != q.shape):
        print("shapes of distributions do not match {} & {} ".format(p.shape, q.shape))
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def load_batch(batch=0):
    spot_samples = {}
    start = 50 * batch
    end = start + 50
    znd=300
    if val :
        if len(v_spots)<end:
            end=len(v_spots)
        for i in range(start,end):
            if ".DS_Store" in v_spots[i]: continue
            spot_ = np.asarray(Image.open(os.path.join(v_spot_root, v_spots[i])).convert('L'))
            spot_samples[v_spots[i]]=spot_

            print("\r" + str(i) + "/" + str(len(v_spots)), end="")
        return spot_samples
    if len(v_spots) < end:
        end = len(spots)
    for i in range(start, end):

        if ".DS_Store" in spots[i]: continue
        spot_ = np.asarray(Image.open(os.path.join(spot_root, spots[i])).convert('L'))
        spot_samples[spots[i]] = spot_
        print("\r" + str(i) + "/" + str(len(spots)), end="")

    return spot_samples

def get_lbp_data():
    lbp_data=load_batch(BATCH).values()
    return lbp_data

def LBP_prep():
    # gather pattern images
    patterns_img = get_lbp_data()

    # gather lbp images
    LBP_img = []
    for img in patterns_img:
        lbp = get_lbp(img)
        LBP_img.append(lbp)

    # gather feature hitograms
    LBP_hist = []
    for lbp in LBP_img:
        h = blocks(lbp)
        hist = feature_hist(h)
     #   LBP_hist.append(get_hist_LBP(lbp))
        LBP_hist.append(hist)

    return patterns_img,LBP_img,LBP_hist

def LBP_one2many(idx,patterns_img,LBP_img,LBP_hist):

    """

    :param idx: index of one to compare to others
    :param patterns_img:
    :param LBP_img:
    :param LBP_hist:
    :return: kullb
    """
    orignal_img = list(patterns_img)[idx]
    original_lbp = LBP_img[idx]
    original_hist = LBP_hist[idx]
    kld_dissimilarity = []
    chi_dissimilarity = []
    for hist in LBP_hist:
        kld_score = kullback_leibler_divergence(hist, original_hist)
        chi_score = chi2_distance(hist,original_hist)
        kld_dissimilarity.append(kld_score)
        chi_dissimilarity.append(chi_score)
    return kld_dissimilarity,chi_dissimilarity

def plot_LBP_encoding():

    patterns_img, LBP_img, LBP_hist= LBP_prep()
    print("time to plot ")
    for spot,lbp in zip(patterns_img,LBP_img):

        fig = plt.figure(figsize=(15, 11))
        plt.subplot(1, 2, 1)
        plt.title(" spot pattern ")
        plt.imshow(spot.astype('uint8'),cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(" LBP encoded spot pattern ")
        plt.imshow(lbp.astype('uint8'), cmap="gray")
        plt.show()
        print("new plot in ")

def plot_LBP_encoding_radius_variation(nb_points,radius):

    idx=6
    patterns_img, _, _= LBP_prep()
    image = list(patterns_img)[idx]

    fig = plt.figure(figsize=(15, 11))
    plt.subplot(1, 5, 1)
    plt.title(" Spot pattern ")
    plt.axis("off")
    plt.imshow(image.astype('uint8'), cmap="gray")
    assert(len(radius)==4 )
    for i,r in enumerate(radius):

        lbp=get_lbp(image,r,nb_points)

        plt.subplot(1, 5, i+2)
        plt.title(" LBP with radius of {} ".format(r))
        plt.axis("off")
        plt.imshow(lbp.astype('uint8'), cmap="gray")
    plt.show()

def plot_LBP_encoding_points_variation(nb_points,radius):

    idx=6
    patterns_img, _, _= LBP_prep()
    image = list(patterns_img)[idx]

    fig = plt.figure(figsize=(15, 11))
    plt.subplot(1, 5, 1)
    plt.title(" Spot pattern ")
    plt.axis("off")
    plt.imshow(image.astype('uint8'), cmap="gray")
    assert(len(nb_points)==4 )
    for i,n in enumerate(nb_points):

        lbp=get_lbp(image,radius,n)

        plt.subplot(1, 5, i+2)
        plt.title(" LBP with {} points".format(n))
        plt.axis("off")
        plt.imshow(lbp.astype('uint8'), cmap="gray")
    plt.show()

def plot_LBP_hist():

    patterns_img, LBP_img, LBP_hist= LBP_prep()
    print("time to plot ")
    for spot,hist in zip(patterns_img,LBP_hist):

        fig = plt.figure(figsize=(15, 11))
        plt.subplot(1, 2, 1)
        plt.title(" spot pattern ")
        plt.imshow(spot.astype('uint8'),cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(" LBP encoded histogram ")
        plt.hist(hist, density=True)
        plt.show()
        #print("new plot in ")

def plot_one2many():

    idx=39
    patterns_img, LBP_img, LBP_hist= LBP_prep()
    orignal_img = list(patterns_img)[idx]
    original_lbp = LBP_img[idx]
    original_hist = LBP_hist[idx]
    kld_scores,chi_scores=LBP_one2many(idx,patterns_img, LBP_img, LBP_hist)

    sorted_scores = sorted(kld_scores)
    min_value = sorted_scores[1]
    min_position = kld_scores.index(min_value)

    max_value = sorted_scores[-1]
    max_position = kld_scores.index(max_value)

    print("kld minimum divergence: {} at pos {}".format(min_value,min_position))
    print("kld maximum divergence: {} at pos {}".format(max_value,max_position))

    sorted_scores = sorted(chi_scores)
    min_value = sorted_scores[1]
    min_position = chi_scores.index(min_value)

    max_value = sorted_scores[-1]
    max_position = chi_scores.index(max_value)

    print("chi minimum divergence: {} at pos {}".format(min_value,min_position))
    print("chi maximum divergence: {} at pos {}".format(max_value,max_position))

    for spot, lbp, hist,kld_score,chi_score  in zip(patterns_img, LBP_img,LBP_hist,kld_scores,chi_scores):

        fig = plt.figure(figsize=(15, 11))
        plt.subplot(2, 3, 1)
        plt.title(" Original spot pattern ")
        plt.axis("off")
        plt.imshow(orignal_img.astype('uint8'),cmap="gray")

        plt.subplot(2, 3, 2)
        plt.title(" Original LBP encoded spot pattern ")
        plt.imshow(original_lbp.astype('uint8'), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title(" Original LBP hist ")
        plt.hist(original_hist, density=True)

        plt.subplot(2, 3, 4)
        plt.title(" Compared spot pattern ")
        plt.imshow(spot.astype('uint8'), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.title(" Compared LBP encoded spot pattern ")
        plt.imshow(lbp.astype('uint8'), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.title(" Compared LBP hist ")
        plt.hist(hist, density=True)

        plt.subplots_adjust(top=0.9)
        plt.suptitle("KLD Score: {}\nCHI Score: {}".format(kld_score, chi_score), fontweight="bold", fontsize=16)

        plt.show()


    fig = plt.figure(figsize=(15, 11))
    plt.subplot(2, 3, 1)
    plt.title(" Original spot pattern ")
    plt.imshow(orignal_img.astype('uint8'),cmap="gray")

    plt.subplot(2, 3, 2)
    plt.title(" Original LBP encoded spot pattern ")
    plt.imshow(original_lbp.astype('uint8'), cmap="gray")

    plt.subplot(2, 3, 3)
    plt.title(" Original LBP hist ")
    plt.hist(original_hist, density=True)

    plt.subplot(2, 3, 4)
    plt.title(" Closest spot pattern ")
    plt.imshow(list(patterns_img)[min_position].astype('uint8'), cmap="gray")

    plt.subplot(2, 3, 5)
    plt.title(" Closest LBP encoded spot pattern ")
    plt.imshow(LBP_img[min_position].astype('uint8'), cmap="gray")

    plt.subplot(2, 3, 6)
    plt.title(" Closest LBP hist ")
    plt.hist(LBP_hist[min_position], density=True)

    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Closest with KLD Score: {kld_score}",fontweight="bold", fontsize=16)

    plt.show()

def plot_one2many_top5():

    idx=8
    patterns_img, LBP_img, LBP_hist= LBP_prep()
    orignal_img = list(patterns_img)[idx]
    original_lbp = LBP_img[idx]
    original_hist = LBP_hist[idx]
    kld_scores,chi_scores=LBP_one2many(idx,patterns_img, LBP_img, LBP_hist)
    top_5_KLD= sorted(kld_scores)[:5]
    top_5_chi= sorted(chi_scores)[:5]

    patterns_img=list(patterns_img)

    fig = plt.figure(figsize=(15, 11))
    plt.subplot(2, 3, 1)
    plt.title(" Original spot pattern ")
    plt.axis("off")
    plt.imshow(orignal_img.astype('uint8'), cmap="gray")

    plt.subplot(2, 3, 2)
    plt.title(" Top 1 similar spot pattern ")
    plt.imshow(patterns_img[kld_scores.index(top_5_KLD[0])].astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title(" Top 2 similar spot pattern ")
    plt.imshow(patterns_img[kld_scores.index(top_5_KLD[1])].astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title(" Top 3 similar spot pattern ")
    plt.imshow(patterns_img[kld_scores.index(top_5_KLD[2])].astype('uint8'), cmap="gray")
    plt.axis("off")


    plt.subplot(2, 3, 5)
    plt.title(" Top 2 similar spot pattern ")
    plt.imshow(patterns_img[kld_scores.index(top_5_KLD[3])].astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title(" Top 2 similar spot pattern ")
    plt.imshow(patterns_img[kld_scores.index(top_5_KLD[4])].astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplots_adjust(top=0.9)
    plt.suptitle("top five kld similar patterns", fontweight="bold", fontsize=16)

    plt.show()


    fig = plt.figure(figsize=(15, 11))
    plt.subplot(2, 3, 1)
    plt.title(" Original spot pattern ")
    plt.axis("off")
    plt.imshow(orignal_img.astype('uint8'), cmap="gray")

    plt.subplot(2, 3, 2)
    plt.title(" Top 1 similar spot pattern ")
    plt.imshow(patterns_img[chi_scores.index(top_5_chi[0])].astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title(" Top 2 similar spot pattern ")
    plt.imshow(patterns_img[chi_scores.index(top_5_chi[1])].astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title(" Top 3 similar spot pattern ")
    plt.imshow(patterns_img[chi_scores.index(top_5_chi[2])].astype('uint8'), cmap="gray")
    plt.axis("off")


    plt.subplot(2, 3, 5)
    plt.title(" Top 2 similar spot pattern ")
    plt.imshow(patterns_img[chi_scores.index(top_5_chi[3])].astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title(" Top 2 similar spot pattern ")
    plt.imshow(patterns_img[chi_scores.index(top_5_chi[4])].astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplots_adjust(top=0.9)
    plt.suptitle("top five chi similar patterns", fontweight="bold", fontsize=16)

    plt.show()

def comparepairLBP():
    path_0="/Users/sarralaksaci/Desktop/28.png"
    path_1="/Users/sarralaksaci/Desktop/28_one.png"
    path_2="/Users/sarralaksaci/Desktop/28_two.png"
    path_3="/Users/sarralaksaci/Desktop/28_three.png"

    path_diff="/Users/sarralaksaci/Desktop/7.png"


    exp_o= np.asarray(Image.open(path_0).convert('L'))
    exp_1 = np.asarray(Image.open(path_1).convert('L'))
    exp_2 = np.asarray(Image.open(path_2).convert('L'))
    exp_3 = np.asarray(Image.open(path_3).convert('L'))

    exp_diff= np.asarray(Image.open(path_diff).convert('L'))

    lbp_o= get_lbp(exp_o)
    hist_o = get_hist_LBP(lbp_o)

    lbp_1= get_lbp(exp_1)
    hist_1 = get_hist_LBP(lbp_1)

    lbp_2= get_lbp(exp_2)
    hist_2 = get_hist_LBP(lbp_2)

    lbp_3 = get_lbp(exp_3)
    hist_3 = get_hist_LBP(lbp_3)

    lbp_diff = get_lbp(exp_diff)
    hist_diff = get_hist_LBP(lbp_diff)

    kld_score_1 = kullback_leibler_divergence(hist_o, hist_1)
    chi_score_1 = chi2_distance(hist_o, hist_1)
    kld_score_2 = kullback_leibler_divergence(hist_o, hist_2)
    chi_score_2 = chi2_distance(hist_o, hist_2)
    kld_score_3 = kullback_leibler_divergence(hist_o, hist_3)
    chi_score_3 = chi2_distance(hist_o, hist_3)

    kld_score_diff = kullback_leibler_divergence(hist_o, hist_diff)
    chi_score_diff = chi2_distance(hist_o, hist_diff)

    print("KLD scores:  one= {} ; two= {} ; three= {} ".format(kld_score_1,kld_score_2,kld_score_3))
    print("Chi scores:  one= {} ; two= {} ; three= {} ".format(chi_score_1,chi_score_2,chi_score_3))

    fig = plt.figure(figsize=(20, 16))
    plt.subplot(1, 5, 1)
    plt.title(" Original spot pattern ")
    plt.axis("off")
    plt.imshow(exp_o.astype('uint8'), cmap="gray")

    plt.subplot(1, 5, 2)
    plt.title(" (a) \n kld score: {:.3f} ; chi score: {:.3f} ".format(kld_score_1,chi_score_1))
    plt.imshow(exp_1.astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.title(" (b) \n kld score: {:.3f} ; chi score: {:.3f} ".format(kld_score_2,chi_score_2))
    plt.axis("off")
    plt.imshow(exp_2.astype('uint8'), cmap="gray")

    plt.subplot(1, 5, 4)
    plt.title("  (c)  \n kld score: {:.3f} ; chi score: {:.3f} ".format(kld_score_3,chi_score_3))
    plt.imshow(exp_3.astype('uint8'), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.title("  (d)  \n kld score: {:.3f} ; chi score: {:.3f} ".format(kld_score_diff, chi_score_diff))
    plt.imshow(exp_diff.astype('uint8'), cmap="gray")
    plt.axis("off")

    #plt.subplots_adjust(top=0.9)
    #plt.suptitle("KLD score : {} \n Chi score: {}".format(kld_score,chi_score), fontweight="bold", fontsize=16)

    plt.show()











