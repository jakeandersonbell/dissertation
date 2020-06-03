import rasterio
import os

parent_path = "D:/Dissertation"

year = '2012'

tile = 'TQ35'

path = parent_path + "/imagery/" + year + "/" + tile + "/" + tile[:2].lower() + "/"

images = [i for i in os.listdir(path) if ".jpg" in i]

with rasterio.open(path + images[0])as dataset:
    print(dataset.bounds)
