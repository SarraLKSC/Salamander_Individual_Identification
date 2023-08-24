import os
import pandas as pd

### Code applied on the data returned by intern Louis after labeling individual salamanders on mandermatcher
def csv_tables(TabFsmSgh,TabFsmSghImg):
    """
        returns the required dataframe based on Mandermatcher manual identification csv
    """
    img_df = pd.read_csv(TabFsmSghImg, sep=";")
    id_df = pd.read_csv(TabFsmSgh, sep=";")

    img_df = img_df.iloc[:, 0:2]
    id_df = id_df.iloc[:, 0:2]
    ided_img = pd.merge(id_df, img_df, on="SghIdf")
    ided_img = ided_img.iloc[:, 1:]
    ided_img = ided_img.sort_values(by="FsmIdf")

    return ided_img

def images_names(path="/Users/sarralaksaci/PycharmProjects/Salamanders_Thesis/data/Images"):
    """
    :param path: path to the ManderMatcher exported images
    :return: root path and list of file names
    """
    imgs=os.listdir(path)
    if ".DS_Store" in imgs:
        imgs.remove(".DS_Store")
    if "desktop.ini" in imgs:
        imgs.remove("desktop.ini")
    return path,imgs


def file_ImgIdf(imgs):
    """
    :param imgs: file names
    :return: dictionary with Image identifier as key and file name as value
    """
    file_idf={}
    for img in imgs:
        nb=extract_ImgIdf(img)
        if nb not in file_idf.keys():
            file_idf[nb]=img
    return file_idf

def extract_ImgIdf(file_name):
    """
    :return: int id of the file name
    """
    return( int(file_name.split(".")[0]))

def rename_IdedFile(file_idf, id_df):
    """
    :param file_idf: dictionary with ImgIdf as key and file name as key
    :param id_df: dataframe for Individual idf and Image idf
    """

    root="/Users/sarralaksaci/PycharmProjects/Salamanders_Thesis/data/Images"
    max_idf=id_df.iloc[:,0].max()
    for i in range(1,max_idf+1):
        tmp = id_df[id_df["FsmIdf"] == i]
        img_ids=list(tmp["ImgIdf"])
        if len(img_ids)>0:
            for j in range(len(img_ids)):
                old_file_name= file_idf[img_ids[j]]
                new_file_name="{}_{}.jpg".format(i,j)
                os.rename(os.path.join(root,old_file_name),os.path.join(root,new_file_name))

def get_indiv_images(idf,TabId,file_idf):
  """
      idf: int identity number of individual
      TabID: pandas dataframe of individual id and image id conbinations
      file_idf: dictionary of image id keys and image files value
      return: list of images files of individual idf
  """
  imgs=[]
  rows=TabId[TabId["FsmIdf"]==idf]
  img_idfs=list(rows["ImgIdf"])
  for img in img_idfs:
    imgs.append(file_idf[img])
  return imgs


def mainETL():
    df=csv_tables("/Users/sarralaksaci/PycharmProjects/Salamanders_Thesis/data/TabFsmSgh.csv","/Users/sarralaksaci/PycharmProjects/Salamanders_Thesis/data/TabFsmSghImg.csv")
    path,imgs=images_names()
    id_dict=file_ImgIdf(imgs)
    print(id_dict)
    rename_IdedFile(id_dict,df)