"""Filtros passa-baixa no domínio da frequência (ideal, Butterworth, gaussiano).

Fórmulas: slides "Filtros no Domínio da Frequência". Assinatura de createPB igual
ao template da aula (Prática_Filtragem_no_Domínio_da_Frequência.ipynb).
"""
import numpy as np

FILTROS = {0: 'Ideal', 1: 'Butterworth', 2: 'Gaussiano'}


def createPB(shape, center, radius, lpType=2, n=2):
    """Máscara H(u,v) passa-baixa.

    lpType: 0 = ideal, 1 = Butterworth (ordem n), 2 = gaussiano. radius = D0.
    D(u,v) é a distância de (u,v) até `center` = (coluna, linha).
    """
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float64)
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2) + np.power(r, 2))                # D(u,v)
    if lpType == 0:      # ideal: 1 dentro do raio D0, 0 fora
        return (d <= radius).astype(np.float64)
    if lpType == 1:      # Butterworth: 1 / (1 + (D/D0)^2n)
        return 1 / (1 + np.power(d / radius, 2 * n))
    return np.exp(-np.power(d, 2) / (2 * radius ** 2))         # gaussiano: e^(-D²/2D0²)


def filtrar_pb(f, lpType, D0, n=2):
    """Aplica o passa-baixa e devolve a imagem filtrada (float) do tamanho de `f`.

    Preenchimento com zeros (P=2M, Q=2N), como nos slides, para que F·H seja
    convolução linear e não circular. Volta ao espaço pela parte REAL: a magnitude
    rebateria os valores negativos do ringing e criaria texto falso.
    """
    M, N = f.shape
    P, Q = 2 * M, 2 * N
    fp = np.zeros((P, Q))
    fp[:M, :N] = f
    F = np.fft.fftshift(np.fft.fft2(fp))                        # espectro centralizado
    H = createPB((P, Q), center=(Q // 2, P // 2), radius=D0, lpType=lpType, n=n)
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))          # G = F·H e volta ao espaço
    return g[:M, :N]


def para_uint8(g):
    """Recorta em [0, 255] (sem normalizar min-max, para manter a mesma escala entre filtros)."""
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


# --- Filtragem espacial auxiliar (usada pelo Canny e pelo SSIM) ---

def convolve2d(image, kernel):
    """`convolve2d` da aula (correlação com preenchimento de zeros), sem os laços:
    cada pixel de saída é a soma da janela vizinha multiplicada pelo kernel."""
    kh, kw = kernel.shape
    padded = np.pad(image.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode='constant')
    janelas = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum('ijkl,kl->ij', janelas, kernel)


def gaussian_kernel(size=5, sigma=1.0):
    """Kernel gaussiano normalizado (aula)."""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)
