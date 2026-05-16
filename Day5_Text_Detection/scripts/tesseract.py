
from PIL import Image
import os
import pytesseract

Data_set = "Dataset"


def text_det_tess(image_path):
    results = {}

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for filename in os.listdir(Data_set):

        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(Data_set, filename)
            print(f"Processing\n: {filename}")
            try:
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img, lang="eng")
                results[filename] = text
                print("OUTPUT:")
                print(text)
            except Exception as err:
                results[filename] = f"Error: {err}"

    return results




"""Printing using this when you need to run the """
# output = text_det_tess(Data_set)
# for file_name, text in output.items():
#     print("\n-------------------")
#     print("FILE:", file_name)
#     print("TEXT:")
#     print(text)