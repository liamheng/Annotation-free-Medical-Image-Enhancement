# https://github.com/SynthyAI/niqe
import math
import numpy as np
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.io
import cv2
import os

alpha_p = np.arange(0.2, 10, 0.001)
alpha_r_p = scipy.special.gamma(2.0 / alpha_p) ** 2 / (scipy.special.gamma(1.0 / alpha_p) * scipy.special.gamma(3. / alpha_p))

def estimate_aggd_params(x):
    x_left = x[x < 0]
    x_right = x[x >= 0]
    stddev_left = math.sqrt((np.sum(x_left ** 2) / (x_left.size)))
    stddev_right = math.sqrt((np.sum(x_right ** 2) / (x_right.size)))

    if stddev_right == 0:
        return 1, 0, 0
    r_hat = np.sum(np.abs(x)) ** 2 / (x.size * np.sum(x ** 2))
    y_hat = stddev_left / stddev_right  # gamma hat
    R_hat = r_hat * (y_hat ** 3 + 1) * (y_hat + 1) / ((y_hat ** 2 + 1) ** 2)

    pos = np.argmin((alpha_r_p - R_hat) ** 2)
    alpha = alpha_p[pos]
    beta_left = stddev_left * math.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha))
    beta_right = stddev_right * math.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha))
    return alpha, beta_left, beta_right


def compute_nss_features(img_norm):
    features = []
    alpha, beta_left, beta_right = estimate_aggd_params(img_norm)
    features.extend([alpha, (beta_left + beta_right) / 2])

    for x_shift, y_shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        img_pair_products = img_norm * np.roll(np.roll(img_norm, y_shift, axis=0), x_shift, axis=1)
        alpha, beta_left, beta_right = estimate_aggd_params(img_pair_products)
        eta = (beta_right - beta_left) * (gamma(2.0 / alpha) / gamma(1.0 / alpha))
        features.extend([alpha, eta, beta_left, beta_right])

    return features


def norm(img, sigma=7 / 6):
    mu = gaussian_filter(img, sigma, mode='nearest', truncate=2.2)
    sigma = np.sqrt(np.abs(gaussian_filter(img * img, sigma, mode='nearest', truncate=2.2) - mu * mu))
    img_norm = (img - mu) / (sigma + 1)
    return img_norm


def niqe(image):
    if image.ndim == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
    
    model_mat = scipy.io.loadmat(os.path.split(__file__)[0] + '/resources/mvg_params.mat')
    model_mu = model_mat['mean']
    model_cov = model_mat['cov']

    features = None
    h, w = img.shape
    quantized_h = max(h // 96, 1) * 96
    quantized_w = max(w // 96, 1) * 96

    quantized_img = img[:quantized_h, :quantized_w]
    img_scaled = quantized_img
    for scale in [1, 2]:
        if scale != 1:
            img_scaled = cv2.resize(quantized_img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        img_norm = norm(img_scaled.astype(float))
        scale_features = []
        block_size = 96 // scale
        for block_col in range(img_norm.shape[0] // block_size):
            for block_row in range(img_norm.shape[1] // block_size):
                block_features = compute_nss_features(
                    img_norm[block_col * block_size:(block_col + 1) * block_size, block_row * block_size:(block_row + 1) * block_size])
                scale_features.append(block_features)

        if features is None:
            features = np.vstack(scale_features)
        else:
            features = np.hstack([features, np.vstack(scale_features)])

    features_mu = np.mean(features, axis=0)
    features_cov = np.cov(features.T)

    pseudoinv_of_avg_cov = np.linalg.pinv((model_cov + features_cov) / 2)
    niqe_quality = math.sqrt((model_mu - features_mu).dot(pseudoinv_of_avg_cov.dot((model_mu - features_mu).T)))

    return niqe_quality
