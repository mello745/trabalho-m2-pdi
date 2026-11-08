"""Métodos comparados: Otsu, k-means e Canny.

Referência: "Prática - Algoritmos de Segmentação.ipynb" do professor (Otsu, k-means e Canny).
Proibido: operações morfológicas.
"""
import numpy as np

from src.filtros import convolve2d


def otsu_threshold(img):
    """Limiar t que maximiza a variância entre classes (implementação da aula)."""
    img = img.astype(np.uint8)
    hist = np.bincount(img.ravel(), minlength=256)
    prob = hist / img.size

    cumulative_prob = np.cumsum(prob)
    cumulative_mean = np.cumsum(prob * np.arange(256))
    global_mean = cumulative_mean[-1]

    numerator = (global_mean * cumulative_prob - cumulative_mean) ** 2
    denominator = cumulative_prob * (1 - cumulative_prob)
    denominator = np.where(denominator == 0, 1e-10, denominator)

    return int(np.argmax(numerator / denominator))


def segmentar_otsu(img):
    """Máscara de texto pelo Otsu. Devolve (mascara, t).

    A classe 1 do Otsu é {0..t}, então texto é `img > t`. Com `>=` (material da aula),
    a imagem binária original vira toda texto: os limiares empatam e o argmax dá t = 0.
    """
    t = otsu_threshold(img)
    return img > t, t


def kmeans_segmentation(img, k=2, max_iter=100, tol=1e-4):
    """k-means nas intensidades (implementação da aula). Devolve (rotulos, centroides).

    Centróides iniciais espaçados entre o mínimo e o máximo, em vez de sorteados:
    com ~93% dos pixels iguais a 0, o sorteio quase sempre repetiria centróides.
    """
    pixels = img.reshape(-1, 1).astype(np.float64)
    centroids = np.linspace(pixels.min(), pixels.max(), k).reshape(k, 1)

    for _ in range(max_iter):
        labels = np.argmin(np.abs(pixels - centroids.T), axis=1)      # atribuição
        new_centroids = centroids.copy()
        for cluster in range(k):                                       # atualização
            cluster_pixels = pixels[labels == cluster]
            if len(cluster_pixels) > 0:
                new_centroids[cluster] = cluster_pixels.mean(axis=0)
        shift = np.linalg.norm(new_centroids - centroids)              # convergência
        centroids = new_centroids
        if shift < tol:
            break

    labels = np.argmin(np.abs(pixels - centroids.T), axis=1)
    return labels.reshape(img.shape), centroids.ravel()


def segmentar_kmeans(img, k):
    """Máscara de texto = cluster de maior centróide. Devolve (mascara, centroides)."""
    labels, centroids = kmeans_segmentation(img, k)
    return labels == np.argmax(centroids), centroids


def sobel_filters(image):
    """Gradiente de Sobel; magnitude normalizada para [0, 255] (aula)."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = convolve2d(image, Kx)
    Gy = convolve2d(image, Ky)
    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / (magnitude.max() + 1e-10) * 255
    direction = np.arctan2(Gy, Gx)
    return magnitude, direction


def non_max_suppression(magnitude, direction):
    """Supressão de não-máximos da aula, vetorizada: mantém o pixel só se ele for o
    máximo entre os dois vizinhos na direção do gradiente (4 faixas de ângulo).

    Alteração: `>` para o vizinho q e `>=` para r (na aula, `>=` nos dois). Numa imagem
    binária os dois pixels da transição têm o mesmo gradiente; com `>=` nos dois, ambos
    ficam, a borda sai com 2 px e encobre as falhas de 1–2 px. O desempate deixa 1 px.
    """
    angle = np.degrees(direction)
    angle[angle < 0] += 180
    m, a = magnitude, angle[1:-1, 1:-1]
    centro = m[1:-1, 1:-1]
    vizinhos = [(m[1:-1, 2:], m[1:-1, :-2]),    # 0°:   (i, j+1) e (i, j-1)
                (m[2:, :-2], m[:-2, 2:]),       # 45°:  (i+1, j-1) e (i-1, j+1)
                (m[2:, 1:-1], m[:-2, 1:-1]),    # 90°:  (i+1, j) e (i-1, j)
                (m[:-2, :-2], m[2:, 2:])]       # 135°: (i-1, j-1) e (i+1, j+1)
    faixa = np.select([(a < 22.5) | (a >= 157.5), a < 67.5, a < 112.5], [0, 1, 2], 3)
    q = np.choose(faixa, [v[0] for v in vizinhos])
    r = np.choose(faixa, [v[1] for v in vizinhos])
    out = np.zeros(m.shape, dtype=np.float32)
    out[1:-1, 1:-1] = np.where((centro > q) & (centro >= r), centro, 0)
    return out


def double_threshold(image, T1=30, T2=100):
    """Pixels fortes (>= T2) e fracos (entre T1 e T2), como na aula."""
    weak, strong = 75, 255
    result = np.zeros(image.shape, dtype=np.uint8)
    result[image >= T2] = strong
    result[(image >= T1) & (image < T2)] = weak
    return result, weak, strong


def hysteresis(image, weak=75, strong=255):
    """Histerese da aula: o pixel fraco vira forte se tiver um vizinho forte, senão é descartado.
    Percorre só os pixels fracos, na mesma ordem (linha a linha) do laço original."""
    h, w = image.shape
    out = image.copy()
    for i, j in np.argwhere(out[1:h-1, 1:w-1] == weak) + 1:
        out[i, j] = strong if np.any(out[i-1:i+2, j-1:j+2] == strong) else 0
    return out


def segmentar_canny(img, T1=30, T2=100):
    """Canny da aula (Sobel, supressão de não-máximos, limiar duplo e histerese), sem a
    suavização gaussiana inicial: nos experimentos, o passa-baixa faz esse papel.
    Devolve o mapa de bordas booleano."""
    magnitude, direction = sobel_filters(img)
    nms = non_max_suppression(magnitude, direction)
    limiar, weak, strong = double_threshold(nms, T1, T2)
    return hysteresis(limiar, weak, strong) == strong
