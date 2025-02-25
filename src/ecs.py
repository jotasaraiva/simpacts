import numpy as np

def ECS(img, band_axis=0, row_axis=1, col_axis=2, smooth_img=None):
    
    assert len(img.shape) == 3, "'img' is not three-dimensional"
    mean_image = img.mean(axis=band_axis)

    if smooth_img is not None:
        assert len(smooth_img.shape) == 3, "'smooth_img' is not three-dimensional"
        assert img.shape == smooth_img.shape, "'img' and 'smooth_img' are different shapes"
        cube = smooth_img.astype(np.float32)
    else:
        cube = img

    bands = cube.shape[band_axis]
    rows = cube.shape[row_axis]
    cols = cube.shape[col_axis]

    mean_image = np.reshape(mean_image, (1, rows * cols))
    D = np.reshape(cube, (bands, rows * cols))
    D = (D - mean_image) ** 2

    d = D.sum(axis=1)

    R = np.empty(mean_image.shape, np.float32)

    for i in range(rows * cols):
        R[0, i] = np.abs(np.corrcoef(d, D[:, i])[0][1])

    R = np.reshape(R, (1, rows, cols))[0]

    return R