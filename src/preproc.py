import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np

def ee_add_amplitude(image, VV = "VV", VH = "VH"):
    amplitude = image\
        .expression('(VV ** 2 + VH ** 2) ** (1 / 2)', {'VV':image.select(VV), 'VH':image.select(VH)})\
        .rename('amplitude')
    return image.addBands(amplitude)

def ee_show_tif(path, band, palette="gray"):
    raster = rio.open(path)
    array = raster.read(band)
    raster.close()
    return plt.imshow(array, cmap=palette)

def normalizer(img, band_axis=0, row_axis=1, col_axis=2):
    bands = img.shape[band_axis]
    rows = img.shape[row_axis]
    cols = img.shape[col_axis]

    img_reduced = np.reshape(img, (bands, rows * cols))
    max_values = np.max(img_reduced, axis=1, keepdims=True)
    min_values = np.min(img_reduced, axis=1, keepdims=True)

    deltas = max_values - min_values
    img_reshaped = (img_reduced - min_values) / deltas
    img_reshaped = np.reshape(img_reshaped, (bands, rows, cols))

    return img_reshaped

def save_with_rio(path, img, template):
    with rio.open(
         path,
         'w',
         driver='GTiff',
         height=img.shape[0],
         width=img.shape[1],
         count=1,
         dtype=img.dtype,
         crs='+proj=latlong',
         transform=template.transform
    ) as dst:
        dst.write(img, 1)
        
    return True