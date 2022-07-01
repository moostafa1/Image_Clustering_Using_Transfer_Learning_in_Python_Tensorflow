import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from fcmeans import FCM
import pandas as pd
import numpy as np
import shutil
import random
import os
from img_cluster import showCluster



X = r"D:\AAL-NTL\MY_WORK\PROJETS\clustering_images\animals"
cls_path = r"D:\AAL-NTL\MY_WORK\PROJETS\clustering_images\clusters"
k = 3
examples = None
models = ["vgg16", "vgg19", "resnet50", "xception", "inceptionv3", "inceptionresnetv2", "densenet", "mobilenetv2"]

imagenet = random.choice(models)
#imagenet = "mobilenetv2"
print("\n\nYour model is: ", imagenet, end='\n\n')





class FCMeans:
    def __init__(self, path=X, n_clusters=k, max_examples=examples, cls_name ="class_", clsPath = cls_path, use_imagenets = imagenet):
        self.path = path
        self.n_clusters = n_clusters
        self.max_examples = max_examples
        self.cls_name = cls_name
        self.clsPath = clsPath
        self.use_imagenets = use_imagenets




    def maxExampls(self):
        try:
            print("\n\n\nStart: maxExampls\n\n")
            paths = os.listdir(self.path)
            #print(paths)
            if self.max_examples == None:
                self.max_examples = len(paths)
            elif self.max_examples > len(paths):
                self.max_examples = len(paths)
            elif self.max_examples < len(paths):
                self.max_examples = self.max_examples
            elif self.max_examples == 0:
                print("\n\nNumber of examples can't be zero.\n\n\n")
                exit()
            random.shuffle(paths)
            self.paths = paths[:self.max_examples]
            print("End: maxExampls\n\n\n")
            return self.paths
        except:
            print("Error: maxExampls", end='\n\n\n')




    def dirRemover(self):
        try:
            print("\n\n\nStart: dirRemover\n\n")
            cls = os.listdir(self.clsPath)
            dirPath = [os.path.join(self.clsPath, c) for c in cls]
            isDir = [os.path.isdir(c) for c in dirPath]
            isDirsPath = [path for i, path in enumerate(dirPath) for j, bool in enumerate(isDir) if i == j and bool == True]
            emptyDir = [shutil.rmtree(f) for f in isDirsPath]
            #delDir = [os.rmdir(d) for d in isDirsPath]
            print("End: dirRemover\n\n\n")
        except:
            print("Error: dirRemover", end='\n\n\n')




    def imgModel(self):
        try:
            print("\n\n\nStart: imgModel\n\n")
            if self.use_imagenets.lower() == "vgg16":
                model = tensorflow.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
            elif self.use_imagenets.lower() == "vgg19":
                model = tensorflow.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
            elif self.use_imagenets.lower() == "resnet50":
                model = tensorflow.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
            elif self.use_imagenets.lower() == "xception":
                model = tensorflow.keras.applications.xception.Xception(include_top=False, weights='imagenet',input_shape=(224,224,3))
            elif self.use_imagenets.lower() == "inceptionv3":
                model = tensorflow.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))
            elif self.use_imagenets.lower() == "inceptionresnetv2":
                model = tensorflow.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
            elif self.use_imagenets.lower() == "densenet":
                model = tensorflow.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=(224,224,3))
            elif self.use_imagenets.lower() == "mobilenetv2":
                model = tensorflow.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3), alpha=1.0, include_top=False, weights='imagenet', pooling=None)
            #self.model = model
            return model
            print("End: imgModel\n\n\n")
        except:
            print("Error: imgModel", end='\n\n\n')




    def imgPreprocess(self, img):
        try:
            #print("\n\n\nStart: imgPreprocess\n\n")
            if self.use_imagenets.lower() == "vgg16":
                img = tensorflow.keras.applications.vgg16.preprocess_input(img)
            elif self.use_imagenets.lower() == "vgg19":
                img = tensorflow.keras.applications.vgg19.preprocess_input(img)
            elif self.use_imagenets.lower() == "resnet50":
                img = tensorflow.keras.applications.resnet50.preprocess_input(img)
            elif self.use_imagenets.lower() == "xception":
                img = tensorflow.keras.applications.xception.preprocess_input(img)
            elif self.use_imagenets.lower() == "inceptionv3":
                img = tensorflow.keras.applications.inception_v3.preprocess_input(img)
            elif self.use_imagenets.lower() == "inceptionresnetv2":
                img = tensorflow.keras.applications.inception_resnet_v2.preprocess_input(img)
            elif self.use_imagenets.lower() == "densenet":
                img = tensorflow.keras.applications.densenet.preprocess_input(img)
            elif self.use_imagenets.lower() == "mobilenetv2":
                img = tensorflow.keras.applications.mobilenet_v2.preprocess_input(img)
            return img
            #print("End: imgPreprocess\n\n\n")
        except:
            print("Error: imgPreprocess", end='\n\n\n')





    def extractFeatures(self):
        try:
            print("\n\n\nStart: extractFeatures\n\n")
            model = self.imgModel()
            #model = VGG19(weights='imagenet', include_top = False, input_shape=(224,224,3))
            img_features = []
            img_name = []
            c=0
            for name in self.paths:
                fname = os.path.join(self.path, name)
                img = load_img(fname, target_size=(224,224,3))
                img = img_to_array(img)
                img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                img = self.imgPreprocess(img)
                feature = model.predict(img)
                feature = feature.flatten()
                #print((feature.shape))
                img_features.append(feature)
                img_name.append(name)
                self.img_features = img_features
                self.img_name = img_name
                c+=1
                print(f"Image: {c}")
            print(f"\nNumber of images: {c} \n\n")
            print("End: extractFeatures\n\n\n")
            return self.img_features, self.img_name
        except:
            print("Error: extractFeatures", end='\n\n\n')





    def fuzzyCMeans(self):
        try:
            print("\n\n\nStart: fuzzyCMeans\n\n")
            self.img_features = np.array(self.img_features, dtype='f')
            #Creating Clusters
            fcm = FCM(n_clusters = self.n_clusters)
            fcm.fit(self.img_features)

            fcm_centers = fcm.centers
            fcm_labels = fcm.predict(self.img_features)

            image_cluster = pd.DataFrame(self.img_name, columns=['image'])
            self.image_cluster = image_cluster
            #print(img_name)
            self.image_cluster["clusterid"] = fcm_labels
            self.image_cluster # 0 denotes cat and 1 denotes dog

            #x_train,x_val,y_train,y_val = train_test_split(self.image_cluster, self.image_cluster["clusterid"], random_state=1)
            print("End: fuzzyCMeans\n\n\n")
            return self.image_cluster
        except:
            print("Error: fuzzyCMeans", end='\n\n\n')




    def clsPathes(self):
        try:
            print("\n\n\nStart: clsPathes\n\n")
            k_lst = []
            cls_lst = []
            for i in range(self.n_clusters):
                k_lst.append(i)
                cls = os.path.join(self.clsPath, self.cls_name+str(i+1))
                cls_lst.append(cls)
                self.k_lst = k_lst
                self.cls_lst = cls_lst
            print("End: clsPathes\n\n\n")
            return self.k_lst, self.cls_lst
        except:
            print("Error: clsPathes", end='\n\n\n')





    def dirMaker(self):
        try:
            print("\n\n\nStart: dirMaker\n\n")
            for cls in self.cls_lst:
                os.mkdir(cls)
            print("End: dirMaker\n\n\n")
        except:
            print("Error: dirMaker", end='\n\n\n')




    def clustering(self):
        try:
            print("\n\n\nStart: clustering\n\n")
            for k in self.k_lst:
                for i in range(len(self.image_cluster)):
                    if self.image_cluster['clusterid'][i]==k:
                        shutil.copy(os.path.join(self.path, self.image_cluster['image'][i]), self.cls_lst[k])
            print("End: clustering\n\n\n")
        except:
            print("Error: clustering", end='\n\n\n')




    def model(self):
        self.maxExampls()
        self.dirRemover()
        self.extractFeatures()
        self.fuzzyCMeans()
        self.clsPathes()
        self.dirMaker()
        self.clustering()
        showCluster(self.clsPath, 20)



if __name__ == "__main__":
    obj = FCMeans()
    obj.model()
