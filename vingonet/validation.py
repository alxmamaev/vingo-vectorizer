import os
import glob
from . import utils
import cv2
import torch
from tqdm import tqdm
from torchvision.transforms import ToTensor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from albumentations import CenterCrop


def validate(model, device, keys_path, queries_path):
    model.eval()

    print("Loading keys images")
    vectors_train, labels_train = [], []
    for image_name in tqdm([i for i in os.listdir(keys_path) if i.endswith(".jpg")]):
        image_path = keys_path + "/" + image_name
        image = utils.load_image(image_path).to(device)
        with torch.no_grad():
            vector = model(image.unsqueeze(0))[0].cpu().numpy()
            vectors_train.append(vector.tolist())
            labels_train.append([image_name[:-4]])

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(vectors_train, labels_train)

    print("Loading queries images")
    vectors_test, labels_test = [], []
    for label in tqdm([i for i in os.listdir(queries_path) if os.path.isdir(queries_path + "/" + i)]):
        for image_path in glob.glob(queries_path + "/" + label + "/*.jpg"):
            for z in [1]:
                image = utils.load_image(image_path, zoom=z).to(device)

                with torch.no_grad():
                    vector = model(image.unsqueeze(0))[0].cpu().numpy()
                    vectors_test.append(vector.tolist())
                    labels_test.append(label)

    pred = knn.predict(vectors_test)
    print("Valudation ready!\n")
    return accuracy_score(pred, labels_test)
