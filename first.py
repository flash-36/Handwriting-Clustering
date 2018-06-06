import cv2
import os, shutil
import numpy as np
from sklearn.cluster import KMeans


source_folder="/media/ujwal/My files/Work/ML/Computer Vision/Handwriting Clustering/Input"
dest_folder="/media/ujwal/My files/Work/ML/Computer Vision/Handwriting Clustering/Output"
cno=10
mindim=600
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
            data.update({filename:np.array(img[0:mindim,0:mindim]).ravel()})#crop every image to mindimxmindim and roll out and add to dict
    return data

def removeFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def createFolder(directory):
    try:
    
        os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

def main():

    placeholder = load_dataset(source_folder)
    data = load_images_from_folder(source_folder)
    

    kmeans = KMeans(n_clusters = cno,random_state = 0).fit(list(data.values()))
    

    fol=[]
    removeFolder(dest_folder) #flush out previous outputs
    for i in range(cno):
        createFolder(dest_folder+"/Cluster "+str(i+1))
        fol.append(dest_folder+"/Cluster "+str(i+1))

    a=list(placeholder.values())
    b=list(placeholder.keys())
    for i in range(len(kmeans.labels_)):
    	cv2.imwrite(fol[kmeans.labels_[i]]+"/"+b[i],a[i])
    	


    	


    












if __name__ == '__main__':
	main()