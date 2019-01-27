import os
import glob
import cv2
import torch
from torchvision.transforms import ToTensor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from albumentations import CenterCrop
from tqdm import tqdm

to_tensor = ToTensor()

def load_image(image_path, crop=True):
    image = cv2.imread(image_path)
    if crop:
        crop_size = min(image.shape[:-1])
        image = CenterCrop(crop_size, crop_size)(image=image)["image"]
    image = cv2.resize(image, (224, 224))
    image = to_tensor(image)

    return image

def validate(model, device, keys_path, queries_path):
    model.eval()

    print("Loading keys images")
    vectors_train, labels_train = [], []
    for image_name in tqdm([i for i in os.listdir(keys_path)
                            if i.endswith(".jpg")]):
        image = load_image(keys_path + "/" + image_name, crop=False).to(device)
        with torch.no_grad():
            vector = model(image.unsqueeze(0))[0].cpu().numpy()
            vectors_train.append(vector.tolist())
            labels_train.append([image_name[:-4]])

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(vectors_train, labels_train)

    print("Loading queries images")
    vectors_test, labels_test = [], []
    for label in tqdm([i for i in os.listdir(queries_path)
                  if os.path.isdir(queries_path + "/" + i)]):
        for image_path in glob.glob(queries_path + "/" + label + "/*.jpg"):
            image = load_image(image_path, crop=True).to(device)

            with torch.no_grad():
                vector = model(image.unsqueeze(0))[0].cpu().numpy()
                vectors_test.append(vector.tolist())
                labels_test.append(label)

    pred = knn.predict(vectors_test)
    print("Valudation ready!\n")
    return accuracy_score(pred, labels_test)
