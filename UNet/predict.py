import os
from PIL import Image
from tqdm import tqdm
from unet import Unet
unet = Unet()
dir_origin_path = "E:\\30 March Morning\saved_data"
dir_save_path = "30M"

name_classes = ["background", "duck"]

if not os.path.exists(dir_save_path):
    os.makedirs(dir_save_path)

img_names = os.listdir(dir_origin_path)
for img_name in tqdm(img_names):
    if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        image_path = os.path.join(dir_origin_path, img_name)
        image = Image.open(image_path)
        r_image = unet.detect_image(image, count=False, name_classes=name_classes)
        r_image.save(os.path.join(dir_save_path, img_name))
