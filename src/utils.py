"""Leitura da imagem, recortes de comparação e exibição."""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recortes usados em TODAS as comparações: nome -> (linha0, linha1, coluna0, coluna1),
# em coordenadas da imagem já sem a moldura.
RECORTES = {
    'A: Historically,': (28, 62, 27, 210),   # letras próximas (ll, ic), pingos do i, vírgula
    'B: programs':      (72, 106, 27, 175),  # g, o, a com espaços internos
    "C: company's":     (246, 282, 27, 190), # muitos espaços internos e o apóstrofo
    'D: "00"':          (291, 325, 389, 456),# aspas coladas nos zeros, dois buracos grandes
}


def carregar_imagem(caminho):
    """Lê a imagem em tons de cinza (uint8)."""
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if img is None:  # o imread não lança erro: devolve None
        raise FileNotFoundError(f'não foi possível ler {caminho}')
    return img


def preparar(bruta):
    """Remove a moldura preta de 1 px e devolve (img, T0, f).

    img: imagem sem moldura (texto preto = 0, fundo branco = 255)
    T0:  máscara booleana do texto original (True = texto)
    f:   texto claro (255) sobre fundo escuro (0), float — entrada dos filtros,
         para que o preenchimento com zeros da FFT equivalha a estender o fundo
    """
    img = bruta[1:-1, 1:-1]
    T0 = img == 0
    return img, T0, T0.astype(np.float64) * 255


def recortar(img, nome):
    """Devolve o pedaço de `img` correspondente a RECORTES[nome]."""
    a, b, c, d = RECORTES[nome]
    return img[a:b, c:d]


def mostrar(ax, im):
    """Exibe com texto escuro sobre fundo claro, como na imagem original."""
    ax.imshow(im.astype(np.uint8) * 255 if im.dtype == bool else im,
              cmap='gray_r', vmin=0, vmax=255, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])


def painel_recortes(linhas, titulo=None, altura_linha=1.05):
    """Grade linhas x recortes. `linhas` = [(rótulo, imagem), ...]. Devolve a figura."""
    larguras = [d - c for _, _, c, d in RECORTES.values()]
    fig, ax = plt.subplots(len(linhas), len(RECORTES), figsize=(15, altura_linha * len(linhas)),
                           gridspec_kw={'width_ratios': larguras}, squeeze=False)
    for i, (rotulo, im) in enumerate(linhas):
        for j, nome in enumerate(RECORTES):
            mostrar(ax[i, j], recortar(im, nome))
            if i == 0:
                ax[i, j].set_title(nome, fontsize=10)
        ax[i, 0].set_ylabel(rotulo, fontsize=8, rotation=0, ha='right', va='center')
    if titulo:
        fig.suptitle(titulo, fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


def grafico_metricas(df, metricas, esperado):
    """Uma métrica por gráfico, em função de D0; cor = filtro, estilo = método."""
    estilos = {'Otsu': '-', 'k-means k=2': ':', 'k-means k=3': '--', 'Canny': '-'}
    cores = {'Ideal': 'tab:red', 'Butterworth': 'tab:blue', 'Gaussiano': 'tab:green'}
    fig, ax = plt.subplots(1, len(metricas), figsize=(4.2 * len(metricas), 3.8))
    base = df[df.filtro == 'Sem filtro']
    for k, met in enumerate(metricas):
        for filtro, cor in cores.items():
            for metodo, ls in estilos.items():
                s = df[(df.filtro == filtro) & (df.metodo == metodo)]
                if not s.empty:
                    ax[k].plot(s.D0, s[met], ls, color=cor, marker='o', ms=3, label=f'{filtro} / {metodo}')
        if met in esperado:
            ax[k].axhline(esperado[met], color='k', ls='--', lw=1)
        ax[k].axhline(base[met].iloc[0], color='gray', lw=1)
        ax[k].set_title(met); ax[k].set_xlabel('D0')
    ax[0].legend(fontsize=7)
    fig.text(0.5, -0.03, 'linha cinza: sem filtragem | tracejado preto: esperado', ha='center', fontsize=9)
    fig.tight_layout()
    return fig
