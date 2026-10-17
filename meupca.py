import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2


def normalizar(X_treino, X_teste=None, metodo='padronizacao'):
    """Ajusta o escalonador SO no treino e aplica nos dois conjuntos."""
    X_tr = np.array(X_treino, dtype=float)

    if metodo == 'padronizacao':
        media = np.mean(X_tr, axis=0)
        dp = np.std(X_tr, axis=0, ddof=1)
        dp = np.where(dp == 0, 1.0, dp)

        X_tr_norm = (X_tr - media) / dp
        if X_teste is not None:
            X_te_norm = (np.array(X_teste, dtype=float) - media) / dp
            return X_tr_norm, X_te_norm
        return X_tr_norm

    elif metodo == 'minmax':
        minimo = np.min(X_tr, axis=0)
        maximo = np.max(X_tr, axis=0)
        delta = maximo - minimo
        delta = np.where(delta == 0, 1.0, delta)

        X_tr_norm = (X_tr - minimo) / delta
        if X_teste is not None:
            X_te_norm = (np.array(X_teste, dtype=float) - minimo) / delta
            return X_tr_norm, X_te_norm
        return X_tr_norm

    else:
        raise ValueError(f"Metodo desconhecido: {metodo}")


def pontuar_atributos(X, y, criterio):
    """Devolve o score de cada atributo. criterio: 'mi', 'anova', 'chi2'."""
    X = np.array(X, dtype=float)
    y = np.array(y)

    if criterio == 'mi':
        scores = mutual_info_classif(X, y, random_state=42)

    elif criterio == 'anova':
        scores, _ = f_classif(X, y)

    elif criterio == 'chi2':
        # chi2 nao aceita numeros negativos
        if np.any(X < 0):
            X = X - np.min(X, axis=0)
        scores, _ = chi2(X, y)

    else:
        raise ValueError(f"Criterio invalido: {criterio}")

    return np.nan_to_num(scores)


def selecionar_topk(scores, k):
    """Devolve os indices dos k atributos mantidos."""
    scores = np.array(scores)

    if k <= 0:
        raise ValueError("k deve ser maior que 0")
    if k > len(scores):
        raise ValueError("k nao pode ser maior que o numero de atributos")

    indexes = np.argsort(scores)[::-1][:k]
    return indexes


def pca_ajustar(X_treino, n_componentes):
    """Centraliza, calcula a covariancia, decompoe e ordena por autovalor.
    Devolve (media, W, variancia_explicada). Ajustar SO no treino."""
    X_treino = np.array(X_treino, dtype=float)

    # 1. centraliza subtraindo a media de cada coluna
    media = np.mean(X_treino, axis=0)
    X_cent = X_treino - media

    # 2. matriz de covariancia
    cov = np.cov(X_cent, rowvar=False)

    # 3. autovalores e autovetores
    autovalores, autovetores = np.linalg.eigh(cov)

    # 4. ordenar do maior para o menor autovalor
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    autovetores = autovetores[:, idx]

    autovalores = np.maximum(autovalores, 0)

    # pega as n componentes escolhidas
    W = autovetores[:, :n_componentes]

    # ajuste de sinal para bater com o sklearn
    for i in range(W.shape[1]):
        if W[np.argmax(np.abs(W[:, i])), i] < 0:
            W[:, i] = -W[:, i]

    # variancia explicada
    variancia_total = np.sum(autovalores)
    variancia_explicada = autovalores[:n_componentes] / variancia_total

    return media, W, variancia_explicada


def pca_projetar(X, media, W):
    """Projeta no subespaco de componentes principais: Z = (X - media) @ W."""
    X = np.array(X, dtype=float)
    X_cent = X - media
    Z = X_cent @ W
    return Z


def cargas_componentes(W, nomes_atributos, n=3):
    """Loadings das n primeiras componentes, para interpretacao no relatorio."""
    cols = [f"PC{i+1}" for i in range(n)]
    df = pd.DataFrame(W[:, :n], index=nomes_atributos, columns=cols)
    return df


if __name__ == "__main__":
    # teste simples
    np.random.seed(42)
    X = np.random.randn(50, 5) * 3 + 10
    nomes = [f"col_{i+1}" for i in range(5)]

    media, W, var_exp = pca_ajustar(X, n_componentes=2)
    Z = pca_projetar(X, media, W)

    print("Shape da projecao:", Z.shape)
    print("Variancia explicada:", var_exp)
    print("\nCargas (loadings):")
    print(cargas_componentes(W, nomes, n=2))