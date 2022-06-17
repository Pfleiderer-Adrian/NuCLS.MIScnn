import math
import os
import itertools
from PIL import Image

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    grid = list(itertools.product(range(0, h - h % d, d), range(0, w - w % d, d)))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

img_path = os.path.join(os.path.join(os.path.join("..", ".."), "0_Public-data-Amgad2019_0.25MPP"), "rgb")
sample_list = os.listdir(img_path)
for sample in sample_list:
    tile(sample, img_path, os.path.join(os.path.join(os.path.join("..", ".."), "BCCS_tiles"), "rgb"), 512)
img_path = os.path.join(os.path.join(os.path.join("..", ".."), "0_Public-data-Amgad2019_0.25MPP"), "mask")
sample_list = os.listdir(img_path)
for sample in sample_list:
    tile(sample, img_path, os.path.join(os.path.join(os.path.join("..", ".."), "BCCS_tiles"), "mask"), 512)
