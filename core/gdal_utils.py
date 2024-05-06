import os
from osgeo import gdal

def extract_pixel_value(file_path, x, y):
    dataset = gdal.Open(file_path)
    bandd = dataset.GetRasterBand(1).ReadAsArray()
    pixel_value = bandd[x, y]
    return pixel_value