from skimage.restoration import denoise_wavelet, cycle_spin, denoise_tv_chambolle, denoise_nl_means, denoise_bilateral
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

def apply_tv(x, w=1.5, show_progress=True):
    xtv = np.ndarray(x.shape)
    t = xtv.shape[0]
    for i in range(t):
        xtv[i, :, :] = denoise_tv_chambolle(
            x[i, :, :],
            weight=w
        )
        if show_progress:
            print("Applying TV: ", str(i+1), "/", str(t), end="\r")
    return xtv

def apply_nlm(x, show_progress = True):
    xnlm = np.ndarray(x.shape)
    t = xnlm.shape[0]
    for i in range(t):
        xnlm[i, :, :] = denoise_nl_means(x[i, :, :])
        if show_progress:
            print("Applying NLM: ", str(i+1), "/", str(t), end="\r")
    return xnlm

def apply_bil(x, show_progress = True):
    xbil = np.ndarray(x.shape)
    t = xbil.shape[0]
    for i in range(t):
        xbil[i, :, :] = denoise_bilateral(x[i, :, :])
        if show_progress:
            print("Applying Bilateral: ", str(i+1), "/", str(t), end="\r")
    return xbil