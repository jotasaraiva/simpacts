from skimage.restoration import denoise_wavelet, cycle_spin, denoise_tv_chambolle
import numpy as np

def denoise_swt(img, wavelet='haar', J=2):
    denoise_kwargs = dict(
        channel_axis=None, wavelet=wavelet, rescale_sigma=True
    )
    
    denoised_img = cycle_spin(
        img,
        func=denoise_wavelet,
        max_shifts=J,
        func_kw=denoise_kwargs,
        channel_axis=None,
        num_workers=1
    )

    return denoised_img

def apply_wavelet(x, show_progress=True):
    xwav = np.ndarray(x.shape)
    t = xwav.shape[0]
    for i in range(t):
        xwav[i, :, :] = denoise_swt(x[i, :, :])
        if show_progress:
            print("Applying Wavelet: ", str(i+1), "/", str(t), end="\r")
    return xwav

def apply_tv(x, w=1.5):
    xtv = np.ndarray(x.shape)
    t = xtv.shape[0]
    for i in range(t):
        xtv[i, :, :] = denoise_tv_chambolle(
            x[i, :, :],
            weight=w
        )
        print(str(i+1)+"/"+str(t), end="\r")
    return xtv