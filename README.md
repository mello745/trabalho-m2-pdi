# Projeto M2 - Filtros Passa-Baixa na Frequência e Segmentação de Caracteres

**Universidade do Vale do Itajaí (UNIVALI)**
Curso: Ciência da Computação
Disciplina: Processamento de Imagens
Professor: Felipe Viel

---

## Integrantes

- Gustavo Corrêa de Mello
- Heloisa Cavalcante Born
- João Paulo Jorge Aimi

---

## Descrição

Comparação de Otsu, k-means e Canny na segmentação de um texto de baixa
resolução com pequenas falhas nos traços dos caracteres, antes e depois da
filtragem passa-baixa no domínio da frequência (ideal, Butterworth e gaussiano).
O objetivo é medir quanto cada filtro reduz as falhas e quando ele causa efeitos
indesejados: apagamento de traços, união de letras ou preenchimento dos espaços
internos dos caracteres.

---

## Estrutura do projeto

```
trabalho-pdi-m2/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                      # Implementações (lógica do projeto)
│   ├── __init__.py
│   ├── utils.py              # Leitura da imagem, recortes e exibição
│   ├── filtros.py            # Passa-baixa ideal, Butterworth e gaussiano (FFT)
│   ├── segmentacao.py        # Otsu, k-means e Canny
│   └── metricas.py           # Métricas de comparação
│
├── notebooks/
│   └── trabalho_m2.ipynb     # Notebook de entrega (executado)
│
├── data/
│   └── texto_falhas.tif      # Imagem fornecida pelo professor
│
├── tests.py                  # Confere as funções contra o código da aula
└── resultados/               # Figuras e CSVs, gerados ao executar o notebook (não versionados)
```

O código fica em `src/` e o notebook apenas importa, executa e exibe. O
relatório é entregue em PDF, fora do repositório.

---

## Como executar

**1. Clonar o repositório**

```bash
git clone <url-do-repositorio>
cd trabalho-pdi-m2
```

**2. Criar e ativar o ambiente virtual**

Linux / macOS:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Instalar as dependências**

```bash
pip install -r requirements.txt
```

**4. Executar o notebook**

```bash
jupyter notebook notebooks/trabalho_m2.ipynb
```

A imagem já está no repositório, em `data/`. O notebook já está salvo executado; ao rodá-lo de novo, as figuras e
as tabelas são gravadas em `resultados/`.

---

## Mapa dos itens do enunciado

| Item | Descrição | Arquivo |
|------|-----------|---------|
| 1 | Filtros passa-baixa ideal, Butterworth e gaussiano | `src/filtros.py` |
| 2 | Otsu, k-means e Canny | `src/segmentacao.py` |
| 3 | Cada método sem filtragem e após a filtragem | `notebooks/trabalho_m2.ipynb` (seções 5 e 6) |
| 4 | Os mesmos recortes em todas as comparações | `src/utils.py` (`RECORTES`) |
| 5 | Redução das falhas | `src/metricas.py` + notebook seção 6 |
| 6 | Apagamento, união de letras, preenchimento | `src/metricas.py` + notebook seções 6 e 7 |
| 7 | Melhor combinação filtro + parâmetros + método | notebook seção 9 e Conclusões |
| 8 | Casos em que a filtragem melhora e piora | notebook seção 8 |
| 9 | Sem operações morfológicas | todo o projeto |
| 10 | Notebook executado com figuras, métricas e conclusões | `notebooks/trabalho_m2.ipynb` |

---

## Restrições do trabalho

Conforme o enunciado:

> Não utilizem operações morfológicas.

**Proibido**
- `cv2.erode`, `cv2.dilate`, `cv2.morphologyEx`, `cv2.getStructuringElement`
- `scipy.ndimage.binary_*` (erosão, dilatação, abertura, fechamento, preenchimento de buracos)
- `skimage.morphology`

Filtros passa-baixa, Otsu, k-means, Canny e métricas (inclusive a convolução
usada no Canny e no SSIM) foram implementados pelo grupo, a partir das práticas
da disciplina. Funções prontas usadas: `numpy.fft` (FFT), `cv2.imread` (leitura)
e `cv2.connectedComponentsWithStats` (só para medir componentes e buracos).

Para conferir que as versões sem laços dão o mesmo resultado que o código da aula:

```bash
python tests.py
```

Todos os integrantes devem saber explicar qualquer parte da implementação.

---

## Dataset

Imagem `Fig0419(a)(text_gaps_of_1_and_2_pixels).tif` (Gonzalez & Woods, 3ª ed.),
do repositório da disciplina:
https://github.com/VielF/ColabProjects/tree/main/dataset/FiltroFrequencia
