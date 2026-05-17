import os
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
# Prepare Data

img2vec = Img2Vec()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "..", "Weather_dataset")
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')


data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    if not os.path.exists(dir_):
        print(f"Folder not found: {dir_}")
        continue

    for category in os.listdir(dir_):
        category_path = os.path.join(dir_, category)
        if not os.path.isdir(category_path):
                continue

        for img_name in os.listdir(category_path):

            img_path_ = os.path.join(dir_, category, img_name)
            try:
                img = Image.open(img_path_).convert("RGB")
                feature_exe = img2vec.get_vec(img)
                features.append(feature_exe)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {img_path_}: {e}")

    data[['training_data', 'validation_data'][j]] = np.array(features)
    data[['training_labels', 'validation_labels'][j]] = np.array(labels)

print(data.keys())


# train Model
model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])



# test performance

y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])
print(score)


# save the model


with open("model2.p", "wb") as f:
     pickle.dump(model, f)
     f.close()