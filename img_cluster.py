import os
import cv2
import numpy as np
from joining_images import stackImages




def showCluster(cluster_dict, img_per_row = 20, wait = 5000):
    #try:
    cls_lst = []
    for i in os.listdir(cluster_dict):
        cls_lst.append(os.path.join(cluster_dict, i))


    all_classes_nested_lst = []
    for c in cls_lst:
        cls = []
        for x in os.listdir(c):
            cls.append(os.path.join(c, x))



        #print(cls, len(cls))
        all_classes_nested_lst.append(cls)  # contains each cluster as a list
    #print(all_classes_nested_lst)


    kernel = r"A_black_image.jpg"   # np.zeros((200,200),np.uint8) # cv2.imread(r"A_black_image.jpg")
    #print(kernel)
    #cv2.imshow('output', kernel)
    #cv2.waitKey(0)


    # to make all clusters divisble by 10 to ake them able to show using stackImages() function
    cls_lens = []
    all_clusters_sliced = []
    for cluster in all_classes_nested_lst:
        cls_lens.append(len(cluster))
        #print(len(cluster))

        flag = True
        while(flag):
            if len(cluster) % img_per_row !=0:
                #print(len(cluster), "not divisable by 10")
                cluster.append(kernel)
            else:
                flag = False

        cluster_slicer = []
        sliced_full_cluster = []
        for imgs in cluster:
            #print(imgs)
            img = cv2.imread(imgs)
            img = cv2.resize(img, (300,300))
            cluster_slicer.append(img)
            if len(cluster_slicer) == img_per_row:
                #print(cluster_slicer, type(cluster_slicer))
                sliced_full_cluster.append(cluster_slicer)    # np.array
                cluster_slicer = []


        all_clusters_sliced.append(sliced_full_cluster)    # np.array


    #all_clusters_sliced = tuple(all_clusters_sliced)
    print(len(all_clusters_sliced))
    print(all_clusters_sliced)

    # return all_clusters_sliced
    for i in range(len(all_clusters_sliced)):
        cluster_num = 'class_'+str(i+1)
        cluster_len = str(cls_lens[i])
        print(cluster_len)
        cluster_name = cluster_num + ",      samples: "+ cluster_len
        #print(type(i))
        if len(all_clusters_sliced[i]) == 0:
            print(f"{cluster_name} directory is empty")
        else:
            imgStack = stackImages(0.2, (all_clusters_sliced[i]))
            cv2.imshow(cluster_name, imgStack)
            cv2.waitKey(wait)

    #except:
    #    print("ERROR: img_cluster")




if __name__ == "__main__":
    X = r"clusters"

    all_clusters_sliced = showCluster(X)

    #print(len(all_clusters_sliced))


    #all_clusters_sliced = np.array(all_clusters_sliced)
    #print(all_clusters_sliced.shape)
