import numpy as np
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
    
# Adapted from https://github.com/al42and/cv_snippets/blob/master/kittler.py
# Copyright (c) 2014, Bob Pepin
# Copyright (c) 2016, Andrey Alekseenko
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

def threshold_ki(img, min_val=0, max_val=1):
    """
    The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin.
    Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
    Paper: Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986).
    """
    h, g = np.histogram(img.ravel(), 265, [min_val, max_val])
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    with np.errstate(invalid='ignore', divide='ignore'):
        sigma_f = np.sqrt(s/c - (m/c)**2)
        sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
        p = c / c[-1]
        v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    return t

def segment2d(x, method='ot'):
    assert method in ('ot', 'ki', 'km'), 'Method needs to be Otsu (ot), Kittler-Illingworth (ki) or KMeans (km)'
    if method == 'ot':
        th = threshold_otsu(x)
        binary = x >= th
        binary = binary.astype('uint8')
        return binary
    elif method == 'ki':
        th = threshold_ki(x)
        binary = x >= th
        binary = binary.astype('uint8')
        return binary
    elif method == 'km':
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x.reshape(-1,1))
        res = kmeans.cluster_centers_[kmeans.labels_].reshape(x.shape)
        return res

def segment_metrics(raster, change, nonchange):
    change = gpd.read_file(change)
    nonchange = gpd.read_file(nonchange)
    res = rio.open(raster)
    
    change_mask, _ = mask(res, change.geometry, crop=True, nodata=2)
    nonchange_mask, _ = mask(res, nonchange.geometry, crop=True, nodata=2)
    res.close()
    
    true_positive = (change_mask == 1).sum()
    false_negative = (change_mask == 0).sum()
    false_positive = (nonchange_mask == 1).sum()
    true_negative = (nonchange_mask == 0).sum()
    
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = (2 * precision * recall)/(precision + recall)
    accuracy = (true_positive + true_negative)/(true_negative + true_positive + false_negative + false_positive)
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}, change_mask, nonchange_mask