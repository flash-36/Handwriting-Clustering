import cv2
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

folder="/home/ujwal/Downloads/dataset"
cno=8
def load_dataset(folder):
    data = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            data.update({filename:img})#make dict key:filename and value:pixel value
    return data



def load_images_from_folder(folder):
    data = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            data.update({filename:np.array(img[0:585,0:585]).ravel()})#crop every image to 585x585 and roll out and add to dict
    return data

def dist(a, b):
    return np.linalg.norm(a-b)


def main():

    placeholder = load_dataset(folder)
    data = load_images_from_folder(folder)
    

    kmeans = KMeans(n_clusters = cno,random_state = 0).fit(list(data.values()))
    fol=[]
    for i in range(cno):
    	fol.append(folder+"/Cluster"+str(i+1))
    a=list(placeholder.values())
    b=list(placeholder.keys())
    for i in range(len(kmeans.labels_)):
    	cv2.imwrite(fol[kmeans.labels_[i]]+"/"+b[i],a[i])
    	


    	


    #for i in images:
    #	print(i.shape)












if __name__ == '__main__':
	main()