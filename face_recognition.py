import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import os
import gc
import time

from matplotlib import pyplot as plt
from skimage.transform import resize
import seaborn as sns

from keras.applications.mobilenet import preprocess_input as preprocess_input_v1, decode_predictions as decode_predictions_v1
from keras.utils import load_img, img_to_array, to_categorical
from keras.applications import MobileNet
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam, AdamW, RMSprop, SGD
from keras.losses import CategoricalCrossentropy
# from keras.metrics import Accuracy
from tensorflow.keras.metrics import F1Score
from keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import wandb
from wandb.keras import WandbCallback


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb_api_jalal")

wandb.login(key=secret_value_0)


ALL_NAMES = "/kaggle/input/lfw-dataset/lfw_allnames.csv"
IMAGES_SRC = "/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled"
PEOPLE = "/kaggle/input/lfw-dataset/people.csv"
PEOPLE_TEST = "/kaggle/input/lfw-dataset/peopleDevTest.csv"
PEOPLE_TRAIN = "/kaggle/input/lfw-dataset/peopleDevTrain.csv"
README = "/kaggle/input/lfw-dataset/lfw_readme.csv"
WORK_DIR = "/kaggle/working/"

df_all = pd.read_csv(ALL_NAMES)
df_all = df_all.sort_values(by="images", ascending=False) # Descending
df_all.describe()


df_all = df_all[df_all["images"] > 70]


unique_name_count = df_all['name'].nunique()

# Display the count of unique values in the "name" column
print("Count of unique names:", unique_name_count)
print("Count of all data:", df_all["images"].sum())


X = []
Y = []

tqdm_all_names = tqdm(list(df_all["name"]), colour="blue")

for name in tqdm_all_names:
    dir_path = os.path.join(IMAGES_SRC, name)
    list_images_name = os.listdir(dir_path)
    
    for image_name in list_images_name:
        images_path = os.path.join(dir_path, image_name)
        rgb_image = load_img(images_path, target_size=(224, 224))
        x = img_to_array(rgb_image)
        x = np.expand_dims(x, axis=0)
        x_v1 = preprocess_input_v1(x)
        X.append(x_v1)
        Y.append(name)

X = np.asarray(X)
Y = np.asarray(Y)

encoder = OneHotEncoder(dtype=np.float32)

# Fit and transform the data
encoded_Y = encoder.fit_transform(Y.reshape(-1, 1))
encoded_Y = encoded_Y.toarray()

encoded_Y.shape
