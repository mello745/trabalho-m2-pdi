"""Métricas para comparar os resultados.

Não há versão limpa da imagem, mas o texto é conhecido: dele saem quantos
componentes e buracos uma segmentação perfeita teria. As métricas estruturais
respondem às perguntas do enunciado (falhas, apagamento, união, preenchimento);
PSNR e SSIM (slides) medem o quanto o filtro alterou a imagem.
Rotular componentes conexos (cv2) só MEDE — não altera imagem, não é morfologia.
"""
import cv2
import numpy as np

from src.filtros import convolve2d, gaussian_kernel

TEXTO = ['Historically, certain computer',
         'programs were written using',
         'only two digits rather than',
         'four to define the applicable',
         'year. Accordingly, the',
         "company's software may",
         'recognize a date using "00"',
         'as 1900 rather than the year',
         '2000.']
PARTES = {'i': 2, 'j': 2, '"': 2}                   # corpo + pingo; dois traços
BURACOS = {c: 1 for c in 'abdegopqADOPQR0469'}      # fonte sem serifa da imagem
BURACOS.update({'B': 2, '8': 2})

AREA_FALHA = 3      # buraco de até 3 px = furo deixado por uma falha de 1–2 px
LARGURA_MAX = 28    # maior caractere isolado (m, w) tem ~24 px; mais largo = letras unidas


def contagens_esperadas(texto=TEXTO):
    """Devolve (componentes, buracos) esperados para o texto."""
    caracteres = ''.join(texto).replace(' ', '')
    return (sum(PARTES.get(c, 1) for c in caracteres),
            sum(BURACOS.get(c, 0) for c in caracteres))


def areas_buracos(mask):
    """Áreas das regiões de fundo que não tocam a borda (vizinhança-4 no fundo)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    borda = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    return np.array([st[j, cv2.CC_STAT_AREA] for j in range(1, n) if j not in borda])


def espessura(mask):
    """Moda do comprimento das corridas horizontais de texto."""
    corridas = []
    for linha in mask.astype(np.int8):
        d = np.diff(np.concatenate([[0], linha, [0]]))
        corridas.extend(np.where(d == -1)[0] - np.where(d == 1)[0])
    return int(np.bincount(corridas).argmax())


def metricas_mascara(mask, ref):
    """Métricas estruturais de uma máscara de texto; `ref` = máscara do texto original."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    buracos = areas_buracos(mask)
    return {
        'componentes': n - 1,
        'buracos_legit': int((buracos > AREA_FALHA).sum()),
        'furos_falha': int((buracos <= AREA_FALHA).sum()),
        'unioes': int((st[1:, cv2.CC_STAT_WIDTH] > LARGURA_MAX).sum()),
        'preservacao_%': 100 * (mask & ref).sum() / ref.sum(),
        'acrescimo_%': 100 * (mask & ~ref).sum() / ref.sum(),
        'espessura_px': espessura(mask),
    }


def metricas_canny(bordas):
    """Segmentos de borda, pontas soltas (pixel com 1 vizinho = contorno aberto) e total de pixels."""
    n, _ = cv2.connectedComponents(bordas.astype(np.uint8), connectivity=8)
    vizinhos = convolve2d(bordas, np.ones((3, 3))) - bordas      # vizinhos de borda na janela 3x3
    return {
        'segmentos': n - 1,
        'pontas_soltas': int((bordas & (vizinhos == 1)).sum()),
        'pixels_borda': int(bordas.sum()),
    }


def escore(m, c_esp, h_esp, furos_orig):
    """Soma dos erros relativos das métricas estruturais (menor é melhor)."""
    return (abs(m['componentes'] - c_esp) / c_esp + abs(m['buracos_legit'] - h_esp) / h_esp
            + m['furos_falha'] / furos_orig + m['unioes'] / c_esp + (1 - m['preservacao_%'] / 100))


def escore_canny(m, contornos_esp, pontas_orig):
    """Erro relativo no nº de contornos + pontas soltas relativas ao original (menor é melhor)."""
    return abs(m['segmentos'] - contornos_esp) / contornos_esp + m['pontas_soltas'] / pontas_orig


def psnr(a, b, L=255):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(L ** 2 / mse)


def ssim(a, b, L=255):
    """SSIM médio com janela gaussiana 11x11, sigma 1.5 (Wang et al.), via convolve2d."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    janela = lambda x: convolve2d(x, gaussian_kernel(11, 1.5))   # médias locais
    mu_a, mu_b = janela(a), janela(b)
    var_a = janela(a * a) - mu_a ** 2
    var_b = janela(b * b) - mu_b ** 2
    cov = janela(a * b) - mu_a * mu_b
    mapa = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2))
    return mapa.mean()
