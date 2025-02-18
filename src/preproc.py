import matplotlib.pyplot as plt
import rasterio as rio

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