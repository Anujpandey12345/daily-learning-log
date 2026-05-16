import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

input_dir = "Dataset"
categories = ['Dataset_empty', 'Dataset_not_empty']


data = []
labels = []

# It.. from all the images

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        image_path = os.path.join(input_dir, category, file)
        try:
            img = imread(image_path)
            if len(img.shape) == 2:
                img = np.stack((img,) * 3, axis=-1)

            img = resize(img, (15, 15, 3))
            data.append(img.flatten())
            labels.append(category_idx)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
        

data = np.asarray(data)
labels = np.asarray(labels)




# Train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


# train classifier 

classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)


# test performance

best_estimator = grid_search.best_estimator_

y_prediction = grid_search.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))





