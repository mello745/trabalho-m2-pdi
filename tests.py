"""Confere que as versões sem laços dão o mesmo resultado que o código da aula.

Uso: python tests.py
"""
import numpy as np

from src.filtros import convolve2d
from src.segmentacao import hysteresis, non_max_suppression, sobel_filters


# --- Versões originais da aula ("Prática - Algoritmos de Segmentação.ipynb"), com laços ---

def convolve2d_aula(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='constant')
    out = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out


def non_max_suppression_aula(magnitude, direction):
    """Laço da aula, com o desempate `> q` adotado em src/segmentacao.py."""
    h, w = magnitude.shape
    out = np.zeros((h, w), dtype=np.float32)
    angle = np.degrees(direction)
    angle[angle < 0] += 180
    for i in range(1, h-1):
        for j in range(1, w-1):
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = magnitude[i, j+1], magnitude[i, j-1]
            elif 22.5 <= a < 67.5:
                q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]
            elif 67.5 <= a < 112.5:
                q, r = magnitude[i+1, j], magnitude[i-1, j]
            else:
                q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]
            if magnitude[i, j] > q and magnitude[i, j] >= r:   # aula: >= q (ver desempate em src)
                out[i, j] = magnitude[i, j]
    return out


def hysteresis_aula(image, weak=75, strong=255):
    h, w = image.shape
    out = image.copy()
    for i in range(1, h-1):
        for j in range(1, w-1):
            if out[i, j] == weak:
                out[i, j] = strong if np.any(out[i-1:i+2, j-1:j+2] == strong) else 0
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (40, 50)).astype(np.float64)
    k = rng.normal(size=(5, 5))
    assert np.allclose(convolve2d(img, k), convolve2d_aula(img, k))

    mag, dire = sobel_filters(img)
    assert np.array_equal(non_max_suppression(mag, dire), non_max_suppression_aula(mag, dire))

    limiar = rng.choice([0, 75, 255], size=(40, 50), p=[0.5, 0.3, 0.2]).astype(np.uint8)
    assert np.array_equal(hysteresis(limiar), hysteresis_aula(limiar))
    print('ok: convolve2d, non_max_suppression e hysteresis iguais às versões da aula')
