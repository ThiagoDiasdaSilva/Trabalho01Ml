import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split

from trat import (
    carregar_dados,
    limpar_dados,
    construir_atributos,
    codificar,
    tratar_outliers_iqr,
    normalizar as normalizar_trat,
    mapa_de_tipos
)
from meupca import pca_ajustar, pca_projetar, pontuar_atributos, selecionar_topk

SEMENTE = 42
np.random.seed(SEMENTE)

PASTA_FIGURAS = "figuras"
PASTA_TABELAS = "tabelas"
os.makedirs(PASTA_FIGURAS, exist_ok=True)
os.makedirs(PASTA_TABELAS, exist_ok=True)

# Mapa ÚNICO de classes: a ordem deve seguir os índices daqui
MAPA_ALVO = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
ROTULOS = ["Dropout", "Enrolled", "Graduate"]


class KNN:
    def __init__(self, k=15, metrica="euclidiana", ponderado=False, p=3):
        self.k = k
        self.metrica = metrica.lower()
        self.ponderado = ponderado
        self.p = p

    def fit(self, X, y):
        self.X_treino = np.asarray(X, dtype=float)
        self.y_treino = np.asarray(y, dtype=int)
        self.classes = np.unique(self.y_treino)
        return self

    def _distancias_em_lote(self, Q):
        if self.metrica in ("euclidiana", "euclidean"):
            q_norm = np.sum(Q**2, axis=1, keepdims=True)
            tr_norm = np.sum(self.X_treino**2, axis=1, keepdims=True).T
            d2 = np.maximum(q_norm + tr_norm - 2.0 * np.dot(Q, self.X_treino.T), 0.0)
            return np.sqrt(d2)

        elif self.metrica in ("manhattan", "cityblock"):
            return np.sum(np.abs(Q[:, np.newaxis, :] - self.X_treino[np.newaxis, :, :]), axis=2)

        elif self.metrica in ("chebyshev", "infinity"):
            return np.max(np.abs(Q[:, np.newaxis, :] - self.X_treino[np.newaxis, :, :]), axis=2)

        elif self.metrica in ("minkowski",):
            diff = np.abs(Q[:, np.newaxis, :] - self.X_treino[np.newaxis, :, :])
            return np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)

        elif self.metrica in ("cosseno", "cosine"):
            q_norm = np.linalg.norm(Q, axis=1, keepdims=True)
            q_norm = np.where(q_norm == 0, 1.0, q_norm)
            tr_norm = np.linalg.norm(self.X_treino, axis=1, keepdims=True).T
            tr_norm = np.where(tr_norm == 0, 1.0, tr_norm)
            sim = np.dot(Q, self.X_treino.T) / (q_norm * tr_norm)
            return np.clip(1.0 - sim, 0.0, 2.0)

        else:
            raise ValueError(f"Métrica de distância desconhecida: {self.metrica}")

    def predict(self, X, tamanho_lote=50):
        X = np.asarray(X, dtype=float)
        n_amostras = X.shape[0]
        predicoes = np.empty(n_amostras, dtype=int)

        for inicio in range(0, n_amostras, tamanho_lote):
            fim = min(inicio + tamanho_lote, n_amostras)
            lote = X[inicio:fim]
            D = self._distancias_em_lote(lote)

            idx_vizinhos = np.argsort(D, axis=1)[:, :self.k]

            for i in range(fim - inicio):
                viz = idx_vizinhos[i]
                classes_viz = self.y_treino[viz]

                if self.ponderado:
                    dist = D[i, viz]
                    pesos = 1.0 / (dist + 1e-10)
                else:
                    pesos = np.ones(self.k)

                votos = np.zeros(len(self.classes))
                for c_idx, classe in enumerate(self.classes):
                    votos[c_idx] = pesos[classes_viz == classe].sum()

                predicoes[inicio + i] = self.classes[np.argmax(votos)]

        return predicoes

    def calcular_matriz_distancias_completa(self, X, tamanho_lote=50):
        X = np.asarray(X, dtype=float)
        n_amostras = X.shape[0]
        matriz_D = np.empty((n_amostras, len(self.X_treino)), dtype=float)

        for inicio in range(0, n_amostras, tamanho_lote):
            fim = min(inicio + tamanho_lote, n_amostras)
            matriz_D[inicio:fim] = self._distancias_em_lote(X[inicio:fim])

        return matriz_D


def matriz_confusao(y_real, y_pred, n_classes=3):
    M = np.zeros((n_classes, n_classes), dtype=int)
    for real, pred in zip(y_real, y_pred):
        M[real, pred] += 1
    return M


def relatorio_metricas(M):
    total = M.sum()
    acuracia = np.trace(M) / total if total > 0 else 0.0

    precisao, revocacao, f1 = [], [], []
    for c in range(M.shape[0]):
        VP = M[c, c]
        FP = M[:, c].sum() - VP
        FN = M[c, :].sum() - VP

        P = VP / (VP + FP) if (VP + FP) > 0 else 0.0
        R = VP / (VP + FN) if (VP + FN) > 0 else 0.0
        F = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0

        precisao.append(P)
        revocacao.append(R)
        f1.append(F)

    return {
        "acuracia": float(acuracia),
        "acuracia_balanceada": float(np.mean(revocacao)),
        "precisao": np.array(precisao),
        "revocacao": np.array(revocacao),
        "f1": np.array(f1),
        "f1_macro": float(np.mean(f1))
    }


def colunas_continuas_de(X_df):
    return [c for c, tipo in mapa_de_tipos.items() if tipo == 'continuos' and c in X_df.columns]


def preparar_referencia_e_alvo(X_ref_df, X_alvo_df, metodo="minmax"):
    """Ajusta outliers (IQR) e normalização SOMENTE na referência (treino)
    e aplica no alvo. Sempre parte dos dados brutos, sem tratamento prévio."""
    cols = colunas_continuas_de(X_ref_df)
    ref, alvo = X_ref_df.copy(), X_alvo_df.copy()
    if cols:
        ref, alvo = tratar_outliers_iqr(ref, alvo, cols)
    if metodo != "nenhuma":
        ref, alvo = normalizar_trat(ref, alvo, metodo=metodo)
    return ref.to_numpy(dtype=float), alvo.to_numpy(dtype=float)


def validacao_cruzada_sem_vazamento(X_df, y_arr, modelo_builder, n_folds=10, n_repeticoes=3,
                                    metodo_norm="minmax", n_pca=None, n_topk=None, semente=SEMENTE):
    y_arr = np.asarray(y_arr, dtype=int)

    scores_acuracia = []
    scores_acuracia_bal = []
    scores_f1_macro = []

    for rep in range(n_repeticoes):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semente + rep * 100)

        for idx_tr, idx_val in cv.split(X_df, y_arr):
            X_tr_arr, X_val_arr = preparar_referencia_e_alvo(
                X_df.iloc[idx_tr], X_df.iloc[idx_val], metodo=metodo_norm
            )
            y_tr_fold = y_arr[idx_tr]
            y_val_fold = y_arr[idx_val]

            if n_pca is not None:
                n_componentes = min(n_pca, X_tr_arr.shape[1], X_tr_arr.shape[0])
                media_pca, W_pca, _ = pca_ajustar(X_tr_arr, n_componentes=n_componentes)
                X_tr_arr = pca_projetar(X_tr_arr, media_pca, W_pca)
                X_val_arr = pca_projetar(X_val_arr, media_pca, W_pca)

            elif n_topk is not None:
                n_atributos = min(n_topk, X_tr_arr.shape[1])
                if n_atributos < X_tr_arr.shape[1]:
                    scores_mi = pontuar_atributos(X_tr_arr, y_tr_fold, criterio='mi')
                    top_indices = selecionar_topk(scores_mi, n_atributos)
                    X_tr_arr = X_tr_arr[:, top_indices]
                    X_val_arr = X_val_arr[:, top_indices]

            modelo = modelo_builder()
            modelo.fit(X_tr_arr, y_tr_fold)
            y_pred_val = modelo.predict(X_val_arr)

            M = matriz_confusao(y_val_fold, y_pred_val, n_classes=3)
            metr = relatorio_metricas(M)

            scores_acuracia.append(metr["acuracia"])
            scores_acuracia_bal.append(metr["acuracia_balanceada"])
            scores_f1_macro.append(metr["f1_macro"])

    return {
        "acuracia_media": float(np.mean(scores_acuracia)),
        "acuracia_std": float(np.std(scores_acuracia)),
        "acuracia_bal_media": float(np.mean(scores_acuracia_bal)),
        "acuracia_bal_std": float(np.std(scores_acuracia_bal)),
        "f1_macro_media": float(np.mean(scores_f1_macro)),
        "f1_macro_std": float(np.std(scores_f1_macro))
    }


def _prever_por_votacao(idx_vizinhos, y_treino, k, n_classes=3):
    classes_viz = y_treino[idx_vizinhos[:, :k]]
    contagens = np.stack([(classes_viz == c).sum(axis=1) for c in range(n_classes)], axis=1)
    return np.argmax(contagens, axis=1)


def selecionar_melhor_knn_cv(X_treino_df, y_treino, n_folds=5, semente=SEMENTE):
    """Grade de (métrica, k) por CV. Pré-processa cada fold UMA vez e calcula
    a matriz de distâncias UMA vez por (fold, métrica), reaproveitando para todos os k."""
    print("\n[Seleção de k e distância por validação cruzada]")

    y_treino = np.asarray(y_treino, dtype=int)
    valores_k = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51]
    metricas = ["euclidiana", "manhattan", "chebyshev", "cosseno"]
    k_max = max(valores_k)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semente)
    folds = []
    for idx_tr, idx_val in cv.split(X_treino_df, y_treino):
        X_tr, X_val = preparar_referencia_e_alvo(
            X_treino_df.iloc[idx_tr], X_treino_df.iloc[idx_val], metodo="minmax"
        )
        folds.append((X_tr, y_treino[idx_tr], X_val, y_treino[idx_val]))

    f1_por_config = {(m, k): [] for m in metricas for k in valores_k}

    for metrica in metricas:
        for X_tr, y_tr_f, X_val, y_val_f in folds:
            modelo = KNN(k=k_max, metrica=metrica).fit(X_tr, y_tr_f)
            D = modelo.calcular_matriz_distancias_completa(X_val)
            idx_viz = np.argsort(D, axis=1)[:, :k_max]
            for k in valores_k:
                y_pred = _prever_por_votacao(idx_viz, y_tr_f, k)
                M = matriz_confusao(y_val_f, y_pred, n_classes=3)
                f1_por_config[(metrica, k)].append(relatorio_metricas(M)["f1_macro"])

    melhor_f1, melhor_k, melhor_metrica = -1.0, 15, "euclidiana"
    for metrica in metricas:
        for k in valores_k:
            f1 = float(np.mean(f1_por_config[(metrica, k)]))
            print(f"  k={k:2d} | {metrica:10s} | F1 Macro={f1:.4f}")
            if f1 > melhor_f1:
                melhor_f1, melhor_k, melhor_metrica = f1, k, metrica

    print(f"\nMelhor configuração: k={melhor_k}, distância={melhor_metrica}, F1 Macro={melhor_f1:.4f}")
    return melhor_k, melhor_metrica, melhor_f1


def experimento_e1_normalizacao(X_tr_df, y_tr, cenario="A", pasta_figuras=PASTA_FIGURAS, pasta_tabelas=PASTA_TABELAS):
    print(f"\n[E1 - Normalização]")
    metodos = ["nenhuma", "minmax", "zscore"]
    nomes = ["Sem Normalização", "Min-Max [0, 1]", "Padronização (Z-Score)"]
    resultados = []

    for met, nome in zip(metodos, nomes):
        res = validacao_cruzada_sem_vazamento(
            X_tr_df, y_tr,
            modelo_builder=lambda: KNN(k=15, metrica="euclidiana", ponderado=False),
            n_folds=5, n_repeticoes=2, metodo_norm=met
        )
        resultados.append({
            "Estratégia": nome,
            "Acurácia_Média": res["acuracia_media"],
            "Acurácia_Std": res["acuracia_std"],
            "F1_Macro_Média": res["f1_macro_media"],
            "F1_Macro_Std": res["f1_macro_std"]
        })
        print(f"  {nome}: Acurácia {res['acuracia_media']*100:.2f}% | F1 Macro {res['f1_macro_media']:.4f}")

    df_e1 = pd.DataFrame(resultados)
    df_e1.to_csv(os.path.join(pasta_tabelas, f"knn_experimento_e1_normalizacao_cenario_{cenario}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(nomes))
    largura = 0.35

    ax.bar(x - largura/2, df_e1["Acurácia_Média"] * 100, largura,
           yerr=df_e1["Acurácia_Std"] * 100, capsize=5, label='Acurácia (%)', color='#3A86FF', alpha=0.85)
    ax.bar(x + largura/2, df_e1["F1_Macro_Média"] * 100, largura,
           yerr=df_e1["F1_Macro_Std"] * 100, capsize=5, label='F1 Macro (x100)', color='#FF006E', alpha=0.85)

    ax.set_title(f"Experimento E1: Efeito da Normalização no k-NN (Cenário {cenario})",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=10, fontweight='bold')
    ax.set_ylabel("Desempenho (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 85)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, loc='upper left')

    fig.tight_layout()
    plt.savefig(os.path.join(pasta_figuras, f"knn_efeito_normalizacao_cenario_{cenario}.png"), bbox_inches='tight')
    plt.close(fig)
    return df_e1


def experimento_e3_distancias_e_k(X_tr_arr, y_tr, X_val_arr, y_val, cenario="A", pasta_figuras=PASTA_FIGURAS, pasta_tabelas=PASTA_TABELAS):
    print(f"\n[E3 - Distâncias e Escolha de k (conjunto de validação)]")
    y_tr = np.asarray(y_tr, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    lista_k = list(range(1, 52, 2))
    metricas_dist = ["euclidiana", "manhattan", "chebyshev", "cosseno"]
    nomes_metricas = ["Euclidiana", "Manhattan", "Chebyshev", "Cosseno"]
    cores = ["#3A86FF", "#8338EC", "#FF006E", "#FB5607"]

    resultados_curvas = {m: {"acuracia": [], "f1_macro": []} for m in metricas_dist}
    melhores_k = {}

    fig_curvas, ax_curvas = plt.subplots(figsize=(10, 6), dpi=300)

    for met, nome, cor in zip(metricas_dist, nomes_metricas, cores):
        modelo = KNN(k=max(lista_k), metrica=met, ponderado=False)
        modelo.fit(X_tr_arr, y_tr)

        D = modelo.calcular_matriz_distancias_completa(X_val_arr)
        idx_vizinhos_todos = np.argsort(D, axis=1)

        acuracias_m, f1_macros_m = [], []
        for k in lista_k:
            y_pred = _prever_por_votacao(idx_vizinhos_todos, y_tr, k)
            metr = relatorio_metricas(matriz_confusao(y_val, y_pred, n_classes=3))
            acuracias_m.append(metr["acuracia"])
            f1_macros_m.append(metr["f1_macro"])

        resultados_curvas[met]["acuracia"] = acuracias_m
        resultados_curvas[met]["f1_macro"] = f1_macros_m

        melhor_idx = int(np.argmax(f1_macros_m))
        melhores_k[nome] = {"k_otimo": lista_k[melhor_idx],
                            "F1_Macro": f1_macros_m[melhor_idx],
                            "Acuracia": acuracias_m[melhor_idx]}

        ax_curvas.plot(lista_k, [f * 100 for f in f1_macros_m],
                       label=f"{nome} (Ótimo: k={lista_k[melhor_idx]}, F1={f1_macros_m[melhor_idx]:.3f})",
                       color=cor, linewidth=2.2, marker='o', markersize=3.5)

    ax_curvas.set_title("Experimento E3: Desempenho em Função de k e da Métrica de Distância",
                        fontsize=12, fontweight='bold', pad=15)
    ax_curvas.set_xlabel("Número de Vizinhos (k)", fontsize=11, fontweight='bold')
    ax_curvas.set_ylabel("F1 Macro (%)", fontsize=11, fontweight='bold')
    ax_curvas.set_xlim(0, 52)
    ax_curvas.grid(True, linestyle='--', alpha=0.3)
    ax_curvas.legend(title="Métrica de Distância", frameon=True, facecolor='white', loc='lower right')

    f1_euc = resultados_curvas["euclidiana"]["f1_macro"]
    ax_curvas.annotate("k = 1: Alta variância\n(sensível a ruído)", xy=(1, f1_euc[0]*100),
                       xytext=(3, f1_euc[0]*100 - 6),
                       arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                       fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF', alpha=0.8))
    ax_curvas.annotate("k grande: Alto viés\n(tende à classe majoritária)",
                       xy=(51, f1_euc[-1]*100), xytext=(32, f1_euc[-1]*100 - 8),
                       arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                       fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF', alpha=0.8))

    fig_curvas.tight_layout()
    plt.savefig(os.path.join(pasta_figuras, f"knn_curva_k_distancias_cenario_{cenario}.png"), bbox_inches='tight')
    plt.close(fig_curvas)

    for met, info in melhores_k.items():
        print(f"  {met}: k={info['k_otimo']} | Acurácia {info['Acuracia']*100:.2f}% | F1 Macro {info['F1_Macro']:.4f}")

    df_e3 = pd.DataFrame([
        {"Métrica": met, "k_Ótimo": info["k_otimo"], "Acurácia (%)": f"{info['Acuracia']*100:.2f}%", "F1_Macro": f"{info['F1_Macro']:.4f}"}
        for met, info in melhores_k.items()
    ])
    df_e3.to_csv(os.path.join(pasta_tabelas, f"knn_experimento_e3_distancias_cenario_{cenario}.csv"), index=False)
    return melhores_k


def experimento_maldicao_dimensionalidade(X_tr_df, y_tr, cenario="A", pasta_figuras=PASTA_FIGURAS, pasta_tabelas=PASTA_TABELAS):
    print(f"\n[Maldição da Dimensionalidade - Seleção vs PCA]")
    dimensoes = [2, 5, 10, 20, 50, None]
    rotulos_dims = ["d=2", "d=5", "d=10", "d=20", "d=50", "Todos"]

    # Mesma normalização (minmax) nos dois métodos, para a comparação ser justa.
    res_selecao, res_pca = [], []

    for d in dimensoes:
        eval_sel = validacao_cruzada_sem_vazamento(
            X_tr_df, y_tr,
            modelo_builder=lambda: KNN(k=15, metrica="euclidiana", ponderado=False),
            n_folds=5, n_repeticoes=2, metodo_norm="minmax", n_topk=d
        )
        res_selecao.append(eval_sel["f1_macro_media"])

        eval_pca = validacao_cruzada_sem_vazamento(
            X_tr_df, y_tr,
            modelo_builder=lambda: KNN(k=15, metrica="euclidiana", ponderado=False),
            n_folds=5, n_repeticoes=2, metodo_norm="minmax", n_pca=d
        )
        res_pca.append(eval_pca["f1_macro_media"])

        d_str = f"d={d}" if d else "Todos"
        print(f"  {d_str:6s} | Seleção F1: {eval_sel['f1_macro_media']:.4f} | PCA F1: {eval_pca['f1_macro_media']:.4f}")

    df_mald = pd.DataFrame({
        "Dimensão": rotulos_dims,
        "F1_Selecao_MI": res_selecao,
        "F1_Projecao_PCA": res_pca
    })
    df_mald.to_csv(os.path.join(pasta_tabelas, f"knn_maldicao_dimensionalidade_cenario_{cenario}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    x = np.arange(len(dimensoes))
    largura = 0.35

    ax.bar(x - largura/2, [f * 100 for f in res_selecao], largura, label='Reduzir Selecionando (Top-k MI)', color='#3A86FF', alpha=0.85)
    ax.bar(x + largura/2, [f * 100 for f in res_pca], largura, label='Reduzir Projetando (PCA Autoral)', color='#8338EC', alpha=0.85)

    ax.set_title(f"Maldição da Dimensionalidade: Seleção de Atributos vs. Projeção PCA no k-NN (Cenário {cenario})",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos_dims, fontsize=10, fontweight='bold')
    ax.set_xlabel("Número de Atributos / Componentes (d)", fontsize=11, fontweight='bold')
    ax.set_ylabel("F1 Macro (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, loc='upper left')

    fig.tight_layout()
    plt.savefig(os.path.join(pasta_figuras, f"knn_maldicao_dimensionalidade_cenario_{cenario}.png"), bbox_inches='tight')
    plt.close(fig)
    return df_mald


def experimento_e4_custo_computacional(X_tr_arr, y_tr, X_te_arr, cenario="A", pasta_tabelas=PASTA_TABELAS):
    print(f"\n[E4 - Custo Computacional]")
    y_tr = np.asarray(y_tr, dtype=int)
    n_inst = 1000
    if len(X_te_arr) >= n_inst:
        X_amostra = X_te_arr[:n_inst]
    else:
        indices = np.random.choice(len(X_te_arr), size=n_inst, replace=True)
        X_amostra = X_te_arr[indices]

    knn_full = KNN(k=15, metrica="euclidiana")
    t0 = time.perf_counter()
    knn_full.fit(X_tr_arr, y_tr)
    t_treino_full = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = knn_full.predict(X_amostra)
    t_pred_full = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    media_pca, W_pca, _ = pca_ajustar(X_tr_arr, n_componentes=10)
    X_tr_pca = pca_projetar(X_tr_arr, media_pca, W_pca)
    t_ajuste_pca = (time.perf_counter() - t0) * 1000

    knn_pca = KNN(k=15, metrica="euclidiana")
    t0 = time.perf_counter()
    knn_pca.fit(X_tr_pca, y_tr)
    t_treino_pca = (time.perf_counter() - t0) * 1000 + t_ajuste_pca

    # A projeção das consultas faz parte do custo de predição com PCA.
    t0 = time.perf_counter()
    X_amostra_pca = pca_projetar(X_amostra, media_pca, W_pca)
    _ = knn_pca.predict(X_amostra_pca)
    t_pred_pca = (time.perf_counter() - t0) * 1000

    ganho_pred = (t_pred_full - t_pred_pca) / t_pred_full * 100

    print(f"  Original ({X_tr_arr.shape[1]} atrib): Treino {t_treino_full:.2f}ms | Predição {t_pred_full:.2f}ms")
    print(f"  Com PCA (d=10): Treino {t_treino_pca:.2f}ms (inclui ajuste do PCA) | Predição {t_pred_pca:.2f}ms ({ganho_pred:.1f}% mais rápido)")

    df_e4 = pd.DataFrame([
        {"Modelo": f"k-NN Original ({X_tr_arr.shape[1]} atrib)", "Tempo_Treino_ms": t_treino_full, "Tempo_Predicao_1000_inst_ms": t_pred_full},
        {"Modelo": "k-NN + PCA (d=10)", "Tempo_Treino_ms": t_treino_pca, "Tempo_Predicao_1000_inst_ms": t_pred_pca}
    ])
    df_e4.to_csv(os.path.join(pasta_tabelas, f"knn_experimento_e4_custo_cenario_{cenario}.csv"), index=False)
    return df_e4


def experimento_e5_modelo_final_teste(X_tr_arr, y_tr, X_te_arr, y_te, k_escolhido=15, metrica_escolhida="euclidiana",
                                      ponderado=False, f1_cv=None, cenario="A", pasta_figuras="figuras", pasta_tabelas="tabelas"):
    print(f"\n[E5 - Avaliação no Teste]")
    y_te = np.asarray(y_te, dtype=int)
    modelo = KNN(k=k_escolhido, metrica=metrica_escolhida, ponderado=ponderado)
    modelo.fit(X_tr_arr, y_tr)
    y_pred = modelo.predict(X_te_arr)

    M = matriz_confusao(y_te, y_pred, n_classes=3)
    metr = relatorio_metricas(M)

    print(f"  Config: k={k_escolhido}, métrica={metrica_escolhida}")
    print(f"  Acurácia: {metr['acuracia'] * 100:.2f}% | Acurácia Bal.: {metr['acuracia_balanceada'] * 100:.2f}% | F1 Macro: {metr['f1_macro']:.4f}")

    if f1_cv is not None:
        diferenca = metr["f1_macro"] - f1_cv
        print(f"  F1 Macro na CV (treino): {f1_cv:.4f} | no teste: {metr['f1_macro']:.4f} | diferença: {diferenca:+.4f}")
        pd.DataFrame([{
            "k": k_escolhido,
            "Métrica": metrica_escolhida,
            "F1_Macro_CV": f1_cv,
            "F1_Macro_Teste": metr["f1_macro"],
            "Diferença": diferenca
        }]).to_csv(os.path.join(pasta_tabelas, f"knn_cv_vs_teste_cenario_{cenario}.csv"), index=False)

    linhas_classe = []
    for i, nome in enumerate(ROTULOS):
        p, r, f = metr['precisao'][i], metr['revocacao'][i], metr['f1'][i]
        print(f"  {nome:8s} -> Precisão: {p:.3f} | Revocação: {r:.3f} | F1: {f:.3f}")
        linhas_classe.append({"Classe": nome, "Precisao": p, "Revocacao": r, "F1": f})

    pd.DataFrame(linhas_classe).to_csv(
        os.path.join(pasta_tabelas, f"knn_metricas_por_classe_cenario_{cenario}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(M, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[f"Pred {r}" for r in ROTULOS],
                yticklabels=[f"Real {r}" for r in ROTULOS],
                annot_kws={"size": 13, "weight": "bold"}, ax=ax)

    ax.set_title(f"Matriz de Confusão do Modelo Final (k-NN, k={k_escolhido}, {metrica_escolhida})\n"
                 f"Cenário {cenario} - Teste Independente", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Predição do Modelo", fontsize=11, fontweight='bold')
    ax.set_ylabel("Classe Real", fontsize=11, fontweight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(pasta_figuras, f"knn_matriz_confusao_final_cenario_{cenario}.png"), bbox_inches='tight')
    plt.close(fig)

    idx_enrolled = MAPA_ALVO["Enrolled"]
    total_enrolled = int(M[idx_enrolled].sum())
    acertos_enrolled = int(M[idx_enrolled, idx_enrolled])
    conf_dropout = int(M[idx_enrolled, MAPA_ALVO["Dropout"]])
    conf_graduate = int(M[idx_enrolled, MAPA_ALVO["Graduate"]])

    print(f"  Classe Enrolled: {acertos_enrolled}/{total_enrolled} acertos ({acertos_enrolled/total_enrolled*100:.1f}%) | "
          f"{conf_dropout} confundidos com Dropout | {conf_graduate} com Graduate")

    return metr


def executar_pipeline_completo(cenario="A"):
    caminho = 'data.csv' if os.path.exists('data.csv') else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data.csv'
    )

    X, y = carregar_dados(caminho, cenario)
    X, y = limpar_dados(X, y)
    X = construir_atributos(X)
    X = codificar(X, mapa_de_tipos)
    y = y.map(MAPA_ALVO).to_numpy(dtype=int)

    X_temp_df, X_te_df, y_temp, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEMENTE, stratify=y
    )
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        X_temp_df, y_temp, test_size=0.2, random_state=SEMENTE, stratify=y_temp
    )

    print("\n" + "="*20)
    print(f"[Cenário {cenario}]")
    print("="*20)
    print(f"\nTreino: {X_tr_df.shape}")
    print(f"Validação: {X_val_df.shape}")
    print(f"Teste: {X_te_df.shape}")

    X_tr_arr, X_val_arr = preparar_referencia_e_alvo(X_tr_df, X_val_df, "minmax")
    _, X_te_arr_tr = preparar_referencia_e_alvo(X_tr_df, X_te_df, "minmax")

    # Experimentos (cada um faz seu próprio pré-processamento por fold).
    experimento_e1_normalizacao(X_tr_df, y_tr, cenario=cenario)
    experimento_maldicao_dimensionalidade(X_tr_df, y_tr, cenario=cenario)

    k_otimo, metrica_otima, f1_cv = selecionar_melhor_knn_cv(X_tr_df, y_tr)

    experimento_e3_distancias_e_k(X_tr_arr, y_tr, X_val_arr, y_val, cenario=cenario)
    experimento_e4_custo_computacional(X_tr_arr, y_tr, X_te_arr_tr, cenario=cenario)

    X_trval_df = pd.concat([X_tr_df, X_val_df])
    y_trval = np.concatenate([y_tr, y_val])
    X_trval_arr, X_te_arr = preparar_referencia_e_alvo(X_trval_df, X_te_df, "minmax")

    experimento_e5_modelo_final_teste(
        X_trval_arr, y_trval,
        X_te_arr, y_te,
        k_escolhido=k_otimo,
        metrica_escolhida=metrica_otima,
        f1_cv=f1_cv,
        cenario=cenario
    )


if __name__ == "__main__":
    executar_pipeline_completo(cenario="A")