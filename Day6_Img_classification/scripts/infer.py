from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model2.p")



with open(model_path, "rb") as f:
    model = pickle.load(f)

img2vec = Img2Vec()

img_path = os.path.join(
    BASE_DIR,
    "..",
    "Weather_dataset",
    "val",
    "shine",
    "Hacker1.jpg"
)


img = Image.open(img_path).convert("RGB")
features = img2vec.get_vec(img)
pred = model.predict([features])
print(f"Prediction of model is : {pred}")