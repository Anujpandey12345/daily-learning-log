from PIL import Image
import os

from easyocr import Reader





Data_path = "Dataset"
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
for filename in os.listdir(Data_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(Data_path, filename)
            print(f"Processing\n: {filename}")
            try:
                text = ''
                reader = Reader(['en'])
                img = Image.open(image_path)
                results = reader.readtext(image_path)
                for result in results:
                    text = text + result[1] + ' '
                text = text[:-1]
                print(text)

            except Exception as err:
                print(f"Processing Error {filename} : {err}")


