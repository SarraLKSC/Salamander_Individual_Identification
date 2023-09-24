# Salamander_Individual_Identification
This repository gathers my work in the context of a masters thesis on the topic of animal individual identification in the case of Fire-Salamanders


## Context 
 <img width="271" alt="Screenshot 2023-09-25 at 00 57 17" src="https://github.com/SarraLKSC/Salamander_Individual_Identification/assets/44327683/0f1906cc-8bd2-4799-a62a-cd3f2c011aac">
 


As of recent years Fire-Salamanders, amphibans present in belgium and across central europe, face the risk of population decline due to the spread of the deadly Bsal fungus in their natural habitat. The presence of this threat and salamander's bio-indicator role makes tracking and monitoring salamander population a task of primse importance. Volunteers in Wallonia and Flanders take part in observation nights during which they photograph all encountered individuals. All encountered individuals are carefully labeled to keep track of the popolation growth and mouvement. Labeling is a tedious task that requires long hours and careful attention. It can be done based on elaborated encoding formulas of softwares such as [mandermatcher](https://jeroenspeybroeck.shinyapps.io/mandermatcher/). Total automation of salamander individual identification based on computer vision techniques is a reseach question that has not yet been answered. This work comes to carry forward the momentum of previous experimental work done by François Duchène in his thesis on the question. His project can be found [here](https://github.com/FrancoisDuchene/salamandres-identification). Precisely, we explored promising handcrafted features for extracted salamander identification pattern in binary format. 

## Dataset 
In the context of this work a dataset was created. This dataset is composed of (i) segmentation data consisting of 1165 annotated images of salamanders and (ii) identification data consisting of 1073 individuals across 1629 images.

## Project 
This project is split in three parts : 
  * Semantic segmentation of the salamander bodies in images. This was achieved through the training of a U-Net.
  * Color segmentation for the extraction of salamanders yellow identification pattern.This was achieved through the use of K-means clustering.
  * Experimentations for the task of individual identificaion. Explored methodes include (i) Local Binary Patterns as a local feature and (ii) a patch-wise approach leveraging image moments.

## Results 

  * Segmentation model reached Dice_loss of 0.19 on test data. (train_data and val_data metric history is displayed bellow)
  * K-means clustering reached Silhouette_score of 0.935.
  * Experiments done with regards to LBP feature descriptor showed that the histograms produced by this method were not distinctive enough to accurately distinguish between the patterns presented.
  * The second method which leveraged image moments as features, in a patch-wise voting system using KNN  [inspired by OCRs ](https://docs.opencv.org/3.4/d8/d4b/tutorial_py_knn_opencv.html) , unvailed the strong discriminative power of image moments for the task (as we can see in the visualization bellow). Using the KNN algorithm, image moments were sufficient to cluster highly similar patter patches together. However the divide-and-conquer strategy of the patch-wise voting approach still carries flaws that introduce false identifications.

    <img width="953" alt="Screenshot 2023-09-25 at 00 20 18" src="https://github.com/SarraLKSC/Salamander_Individual_Identification/assets/44327683/a1ee0a18-b3d8-4ba3-90af-449e2a90cb4a">
    
   <img width="540" alt="momentsViz" src="https://github.com/SarraLKSC/Salamander_Individual_Identification/assets/44327683/8c857ae6-8734-42dc-aa9a-35ea6a44821b">

   <img width="535" alt="momentsviz2" src="https://github.com/SarraLKSC/Salamander_Individual_Identification/assets/44327683/3a980b68-a119-4465-a613-65ba39707da8">
