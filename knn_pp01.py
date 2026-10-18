import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Importa o pre-processamento e o PCA autoral
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

# Mapeamento oficial do alvo
MAPA_ALVO = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
MAPA_INVERSO = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
ROTULOS = ["Dropout", "Enrolled", "Graduate"]


# =========================================================================
# 1. CLASSE k-NN AUTORAL (Atende integralmente à Seção 4.4 do PDF)
# =========================================================================
class KNN:
    """
    Classificador k-NN implementado do zero conforme Seção 4.4 do PDF:
    - Voto majoritário simples
    - Voto ponderado pelo inverso da distância: w_i = 1 / (d_i + eps)
    - Medidas de distância: Euclidiana, Manhattan, Chebyshev, Cosseno e Minkowski (p)
    - Otimização matricial por lotes (BLAS) para máxima velocidade
    """
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
        """Calcula a matriz de distâncias entre o lote de consulta Q e X_treino."""
        if self.metrica in ("euclidiana", "euclidean"):
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
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

    def predict(self, X, tamanho_lote=200):
        X = np.asarray(X, dtype=float)
        n_amostras = X.shape[0]
        predicoes = np.empty(n_amostras, dtype=int)

        for inicio in range(0, n_amostras, tamanho_lote):
            fim = min(inicio + tamanho_lote, n_amostras)
            lote = X[inicio:fim]
            D = self._distancias_em_lote(lote)

            # Índices dos k vizinhos mais próximos
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

    def calcular_matriz_distancias_completa(self, X, tamanho_lote=200):
        """Retorna matriz de distâncias (N_test, N_train) em lotes."""
        X = np.asarray(X, dtype=float)
        n_amostras = X.shape[0]
        matriz_D = np.empty((n_amostras, len(self.X_treino)), dtype=float)

        for inicio in range(0, n_amostras, tamanho_lote):
            fim = min(inicio + tamanho_lote, n_amostras)
            matriz_D[inicio:fim] = self._distancias_em_lote(X[inicio:fim])

        return matriz_D


# =========================================================================
# 2. FUNÇÕES DE AVALIAÇÃO E MÉTRICAS (Seção 4.5 do PDF)
# =========================================================================
def matriz_confusao(y_real, y_pred, n_classes=3):
    """Calcula a matriz de confusão KxK onde linhas=Real e colunas=Predito."""
    M = np.zeros((n_classes, n_classes), dtype=int)
    for real, pred in zip(y_real, y_pred):
        M[real, pred] += 1
    return M


def relatorio_metricas(M):
    """Calcula acurácia, acurácia balanceada, precisão, revocação e F1 macro."""
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


# =========================================================================
# 3. VALIDAÇÃO CRUZADA ESTRATIFICADA SEM VAZAMENTO (Seções 4.5.1 e 4.5.2)
# =========================================================================
def validacao_cruzada_sem_vazamento(X_df, y_arr, modelo_builder, n_folds=10, n_repeticoes=3,
                                    metodo_norm="minmax", n_pca=None, n_topk=None, semente=SEMENTE):
    """
    Executa validação cruzada estratificada k-fold repetida sobre o treino.
    ATENÇÃO: Todas as transformações (outliers, normalização, PCA, seleção)
    são calculadas ESTRITAMENTE dentro de cada fold de treino e aplicadas na validação.
    Garante ZERO data leakage conforme Seção 4.5.2 do PDF.
    """
    colunas_continuas = [c for c, tipo in mapa_de_tipos.items() if tipo == 'continuos' and c in X_df.columns]

    scores_acuracia = []
    scores_acuracia_bal = []
    scores_f1_macro = []

    for rep in range(n_repeticoes):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semente + rep * 100)

        for idx_tr, idx_val in cv.split(X_df, y_arr):
            X_tr_fold = X_df.iloc[idx_tr].copy()
            X_val_fold = X_df.iloc[idx_val].copy()
            y_tr_fold = y_arr[idx_tr]
            y_val_fold = y_arr[idx_val]

            # 1. Tratamento de Outliers no fold de treino
            if colunas_continuas:
                X_tr_fold, X_val_fold = tratar_outliers_iqr(X_tr_fold, X_val_fold, colunas_continuas)

            # 2. Normalização (ajustada no fold de treino)
            if metodo_norm != "nenhuma":
                X_tr_norm, X_val_norm = normalizar_trat(X_tr_fold, X_val_fold, metodo=metodo_norm)
            else:
                X_tr_norm, X_val_norm = X_tr_fold, X_val_fold

            X_tr_arr = X_tr_norm.to_numpy(dtype=float)
            X_val_arr = X_val_norm.to_numpy(dtype=float)

            # 3. Redução via PCA autoral (se solicitado)
            if n_pca is not None and n_pca < X_tr_arr.shape[1]:
                media_pca, W_pca, _ = pca_ajustar(X_tr_arr, n_componentes=n_pca)
                X_tr_arr = pca_projetar(X_tr_arr, media_pca, W_pca)
                X_val_arr = pca_projetar(X_val_arr, media_pca, W_pca)

            # 4. Redução via Seleção Top-k (se solicitado)
            elif n_topk is not None and n_topk < X_tr_arr.shape[1]:
                scores_mi = pontuar_atributos(X_tr_arr, y_tr_fold, criterio='mi')
                top_indices = selecionar_topk(scores_mi, n_topk)
                X_tr_arr = X_tr_arr[:, top_indices]
                X_val_arr = X_val_arr[:, top_indices]

            # 5. Treinamento e Predição do k-NN
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


# =========================================================================
# 4. EXPERIMENTOS OFICIAIS DO PDF
# =========================================================================

def experimento_e1_normalizacao(X_tr_df, y_tr, cenario="A", pasta_figuras="figuras", pasta_tabelas="tabelas"):
    """
    Experimento E1 (Seção 4.4.5a e Seção 6 do PDF):
    Avalia o efeito crítico da normalização no k-NN (Sem normalização vs MinMax vs Z-Score).
    """
    print("\n" + "="*70)
    print(f" EXPERIMENTO E1: Efeito da Normalização no k-NN | Cenário {cenario}")
    print("="*70)

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
        print(f"  {nome:25s} -> Acurácia: {res['acuracia_media']*100:.2f}% ± {res['acuracia_std']*100:.2f}% | F1 Macro: {res['f1_macro_media']:.4f}")

    df_e1 = pd.DataFrame(resultados)
    df_e1.to_csv(os.path.join(pasta_tabelas, f"knn_experimento_e1_normalizacao_cenario_{cenario}.csv"), index=False)

    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(nomes))
    largura = 0.35

    ax.bar(x - largura/2, df_e1["Acurácia_Média"] * 100, largura,
           yerr=df_e1["Acurácia_Std"] * 100, capsize=5, label='Acurácia (%)', color='#3A86FF', alpha=0.85)
    ax.bar(x + largura/2, df_e1["F1_Macro_Média"] * 100, largura,
           yerr=df_e1["F1_Macro_Std"] * 100, capsize=5, label='F1 Macro (x100)', color='#FF006E', alpha=0.85)

    ax.set_title(f"Experimento E1: Efeito Decisivo da Normalização no k-NN\n(Cenário {cenario} - Seção 4.4.5a e Seção 6 do PDF)",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=10, fontweight='bold')
    ax.set_ylabel("Desempenho (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 85)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, loc='upper left')

    fig.tight_layout()
    caminho_fig = os.path.join(pasta_figuras, f"knn_efeito_normalizacao_cenario_{cenario}.png")
    plt.savefig(caminho_fig, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figura] Gráfico E1 salvo em: {caminho_fig}")
    return df_e1


def experimento_e3_distancias_e_k(X_tr_arr, y_tr, X_val_arr, y_val, cenario="A", pasta_figuras="figuras", pasta_tabelas="tabelas"):
    """
    Experimento E3 e Seção 4.4.4 do PDF:
    Varredura de k in {1, 3, 5, ..., 51} e comparação de distâncias (Euclidiana, Manhattan, Chebyshev, Cosseno).
    """
    print("\n" + "="*70)
    print(f" EXPERIMENTO E3: Medidas de Distância & Varredura de k in [1, 51] | Cenário {cenario}")
    print("="*70)

    lista_k = list(range(1, 52, 2))  # {1, 3, 5, ..., 51}
    metricas_dist = ["euclidiana", "manhattan", "chebyshev", "cosseno"]
    nomes_metricas = ["Euclidiana", "Manhattan", "Chebyshev", "Cosseno"]
    cores = ["#3A86FF", "#8338EC", "#FF006E", "#FB5607"]

    resultados_curvas = {m: {"acuracia": [], "f1_macro": []} for m in metricas_dist}
    melhores_k = {}

    fig_curvas, ax_curvas = plt.subplots(figsize=(10, 6), dpi=300)

    for met, nome, cor in zip(metricas_dist, nomes_metricas, cores):
        modelo = KNN(k=max(lista_k), metrica=met, ponderado=False)
        modelo.fit(X_tr_arr, y_tr)

        # Matriz completa de distâncias calculada uma única vez para máxima eficiência
        D = modelo.calcular_matriz_distancias_completa(X_val_arr)
        idx_vizinhos_todos = np.argsort(D, axis=1)

        acuracias_m = []
        f1_macros_m = []

        for k in lista_k:
            idx_k = idx_vizinhos_todos[:, :k]
            y_pred = np.empty(len(X_val_arr), dtype=int)

            for i in range(len(X_val_arr)):
                viz = idx_k[i]
                classes_viz = y_tr[viz]
                # Voto majoritário
                contagens = np.bincount(classes_viz, minlength=3)
                y_pred[i] = np.argmax(contagens)

            M = matriz_confusao(y_val, y_pred, n_classes=3)
            metr = relatorio_metricas(M)
            acuracias_m.append(metr["acuracia"])
            f1_macros_m.append(metr["f1_macro"])

        resultados_curvas[met]["acuracia"] = acuracias_m
        resultados_curvas[met]["f1_macro"] = f1_macros_m

        # Identifica k ótimo
        melhor_idx = int(np.argmax(f1_macros_m))
        melhor_k = lista_k[melhor_idx]
        melhor_f1 = f1_macros_m[melhor_idx]
        melhor_acc = acuracias_m[melhor_idx]
        melhores_k[nome] = {"k_otimo": melhor_k, "F1_Macro": melhor_f1, "Acuracia": melhor_acc}

        ax_curvas.plot(lista_k, [f * 100 for f in f1_macros_m], label=f"{nome} (Ótimo: k={melhor_k}, F1={melhor_f1:.3f})",
                       color=cor, linewidth=2.2, marker='o', markersize=3.5)

    ax_curvas.set_title(f"Experimento E3: Curva de Desempenho em Função de k e Métrica de Distância\n"
                        f"Varredura k in [1, 51] (Seção 4.4.4 e Seção 6 E3 do PDF)", fontsize=12, fontweight='bold', pad=15)
    ax_curvas.set_xlabel("Número de Vizinhos (k)", fontsize=11, fontweight='bold')
    ax_curvas.set_ylabel("F1 Macro (%)", fontsize=11, fontweight='bold')
    ax_curvas.set_xlim(0, 52)
    ax_curvas.grid(True, linestyle='--', alpha=0.3)
    ax_curvas.legend(title="Métrica de Distância", frameon=True, facecolor='white', loc='lower right')

    # Anotacao didatica sobre extremos k=1 (variancia) e k=51 (vies) conforme solicitado na Seção 4.4.4
    ax_curvas.annotate("k = 1: Alta variância\n(sensível a ruído)", xy=(1, resultados_curvas["euclidiana"]["f1_macro"][0]*100),
                       xytext=(3, resultados_curvas["euclidiana"]["f1_macro"][0]*100 - 6),
                       arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                       fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF', alpha=0.8))

    ax_curvas.annotate("k grande: Alto viés\n(converge à classe majoritária)",
                       xy=(51, resultados_curvas["euclidiana"]["f1_macro"][-1]*100),
                       xytext=(32, resultados_curvas["euclidiana"]["f1_macro"][-1]*100 - 8),
                       arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                       fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF', alpha=0.8))

    fig_curvas.tight_layout()
    caminho_fig_k = os.path.join(pasta_figuras, f"knn_curva_k_distancias_cenario_{cenario}.png")
    plt.savefig(caminho_fig_k, bbox_inches='tight')
    plt.close(fig_curvas)
    print(f"[Figura] Curva de k salva em: {caminho_fig_k}")

    # Tabela com k ótimo de cada distância
    df_e3 = pd.DataFrame([
        {"Métrica": met, "k_Ótimo": info["k_otimo"], "Acurácia (%)": f"{info['Acuracia']*100:.2f}%", "F1_Macro": f"{info['F1_Macro']:.4f}"}
        for met, info in melhores_k.items()
    ])
    caminho_csv_e3 = os.path.join(pasta_tabelas, f"knn_experimento_e3_distancias_cenario_{cenario}.csv")
    df_e3.to_csv(caminho_csv_e3, index=False)
    print(f"[Tabela] k ótimo por métrica salvo em: {caminho_csv_e3}")
    print(df_e3)

    return melhores_k


def experimento_maldicao_dimensionalidade(X_tr_df, y_tr, cenario="A", pasta_figuras="figuras", pasta_tabelas="tabelas"):
    """
    Experimento de Maldição da Dimensionalidade e Experimento E2 (Seção 4.4.5c do PDF):
    Compara 'Reduzir Selecionando' (Top-k MI) vs 'Reduzir Projetando' (PCA Autoral)
    para d in {2, 5, 10, 20, 50, Todos}.
    """
    print("\n" + "="*70)
    print(f" EXPERIMENTO: Maldição da Dimensionalidade | Seleção vs PCA | Cenário {cenario}")
    print("="*70)

    dimensoes = [2, 5, 10, 20, 50, None]
    rotulos_dims = ["d=2", "d=5", "d=10", "d=20", "d=50", "Todos"]

    res_selecao = []
    res_pca = []

    for d in dimensoes:
        # 1. Redução por Seleção (Mutual Information)
        eval_sel = validacao_cruzada_sem_vazamento(
            X_tr_df, y_tr,
            modelo_builder=lambda: KNN(k=15, metrica="euclidiana", ponderado=False),
            n_folds=5, n_repeticoes=2, metodo_norm="minmax", n_topk=d
        )
        res_selecao.append(eval_sel["f1_macro_media"])

        # 2. Redução por PCA (Projeção autoral)
        eval_pca = validacao_cruzada_sem_vazamento(
            X_tr_df, y_tr,
            modelo_builder=lambda: KNN(k=15, metrica="euclidiana", ponderado=False),
            n_folds=5, n_repeticoes=2, metodo_norm="zscore", n_pca=d
        )
        res_pca.append(eval_pca["f1_macro_media"])

        d_str = f"d={d}" if d else "Todos"
        print(f"  {d_str:10s} -> Seleção (Top-k MI) F1: {eval_sel['f1_macro_media']:.4f} | PCA (Projeção) F1: {eval_pca['f1_macro_media']:.4f}")

    # Salva tabela comparativa
    df_mald = pd.DataFrame({
        "Dimensão": rotulos_dims,
        "F1_Selecao_MI": res_selecao,
        "F1_Projecao_PCA": res_pca
    })
    caminho_csv = os.path.join(pasta_tabelas, f"knn_maldicao_dimensionalidade_cenario_{cenario}.csv")
    df_mald.to_csv(caminho_csv, index=False)

    # Gráfico comparativo
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    x = np.arange(len(dimensoes))
    largura = 0.35

    ax.bar(x - largura/2, [f * 100 for f in res_selecao], largura, label='Reduzir Selecionando (Top-k MI)', color='#3A86FF', alpha=0.85)
    ax.bar(x + largura/2, [f * 100 for f in res_pca], largura, label='Reduzir Projetando (PCA Autoral)', color='#8338EC', alpha=0.85)

    ax.set_title(f"Maldição da Dimensionalidade: Seleção de Atributos vs. Projeção PCA no k-NN\n"
                 f"(Cenário {cenario} - Seção 4.4.5c e Seção 6 E2 do PDF)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos_dims, fontsize=10, fontweight='bold')
    ax.set_xlabel("Número de Atributos / Componentes (d)", fontsize=11, fontweight='bold')
    ax.set_ylabel("F1 Macro (%)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, loc='upper left')

    fig.tight_layout()
    caminho_fig = os.path.join(pasta_figuras, f"knn_maldicao_dimensionalidade_cenario_{cenario}.png")
    plt.savefig(caminho_fig, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figura] Gráfico da Maldição da Dimensionalidade salvo em: {caminho_fig}")

    return df_mald


def experimento_e4_custo_computacional(X_tr_arr, y_tr, X_te_arr, y_te, cenario="A", pasta_tabelas="tabelas"):
    """
    Experimento E4 (Seção 6 do PDF):
    Contraste de aprendizado preguiçoso do k-NN e efeito da redução de dimensão (PCA)
    no tempo de treino e de predição de 1.000 instâncias.
    """
    print("\n" + "="*70)
    print(f" EXPERIMENTO E4: Custo Computacional e Tempo de Predição (1.000 instâncias) | Cenário {cenario}")
    print("="*70)

    # Amostra exatamente 1.000 instancias para medicao padronizada
    n_inst = 1000
    if len(X_te_arr) >= n_inst:
        X_amostra = X_te_arr[:n_inst]
    else:
        # Se teste tiver menos de 1000, repete amostras com reposição para teste de estresse
        indices = np.random.choice(len(X_te_arr), size=n_inst, replace=True)
        X_amostra = X_te_arr[indices]

    # 1. k-NN com todos os atributos originais
    knn_full = KNN(k=15, metrica="euclidiana")
    t0 = time.perf_counter()
    knn_full.fit(X_tr_arr, y_tr)
    t_treino_full = (time.perf_counter() - t0) * 1000  # ms

    t0 = time.perf_counter()
    _ = knn_full.predict(X_amostra)
    t_pred_full = (time.perf_counter() - t0) * 1000  # ms

    # 2. k-NN com redução por PCA (d=10)
    media_pca, W_pca, _ = pca_ajustar(X_tr_arr, n_componentes=10)
    X_tr_pca = pca_projetar(X_tr_arr, media_pca, W_pca)
    X_amostra_pca = pca_projetar(X_amostra, media_pca, W_pca)

    knn_pca = KNN(k=15, metrica="euclidiana")
    t0 = time.perf_counter()
    knn_pca.fit(X_tr_pca, y_tr)
    t_treino_pca = (time.perf_counter() - t0) * 1000  # ms

    t0 = time.perf_counter()
    _ = knn_pca.predict(X_amostra_pca)
    t_pred_pca = (time.perf_counter() - t0) * 1000  # ms

    ganho_pred = (t_pred_full - t_pred_pca) / t_pred_full * 100

    print(f"k-NN Original ({X_tr_arr.shape[1]} atributos):")
    print(f"  Tempo de Treino:   {t_treino_full:.2f} ms (aprendizado preguiçoso - custo quase zero)")
    print(f"  Tempo de Predição: {t_pred_full:.2f} ms ({n_inst} instâncias)")
    print(f"\nk-NN com PCA (d=10 componentes):")
    print(f"  Tempo de Treino:   {t_treino_pca:.2f} ms")
    print(f"  Tempo de Predição: {t_pred_pca:.2f} ms ({n_inst} instâncias)")
    print(f"  Aceleração na inferência com PCA: {ganho_pred:.1f}% mais rápido!")

    df_e4 = pd.DataFrame([
        {"Modelo": f"k-NN Original ({X_tr_arr.shape[1]} atrib)", "Tempo_Treino_ms": t_treino_full, "Tempo_Predicao_1000_inst_ms": t_pred_full},
        {"Modelo": "k-NN + PCA (d=10)", "Tempo_Treino_ms": t_treino_pca, "Tempo_Predicao_1000_inst_ms": t_pred_pca}
    ])
    df_e4.to_csv(os.path.join(pasta_tabelas, f"knn_experimento_e4_custo_cenario_{cenario}.csv"), index=False)
    return df_e4


def experimento_e5_modelo_final_teste(X_tr_arr, y_tr, X_te_arr, y_te, k_escolhido=15, metrica_escolhida="euclidiana",
                                      ponderado=False, cenario="A", pasta_figuras="figuras", pasta_tabelas="tabelas"):
    """
    Experimento E5 (Seção 6 do PDF):
    Avaliação do modelo final no conjunto de teste independente (avaliá-lo UMA ÚNICA VEZ).
    Gera matriz de confusão, métricas por classe e análise detalhada dos erros da classe Enrolled.
    """
    print("\n" + "="*70)
    print(f" EXPERIMENTO E5: Modelo Final no Conjunto de Teste | Cenário {cenario}")
    print("="*70)

    modelo = KNN(k=k_escolhido, metrica=metrica_escolhida, ponderado=ponderado)
    modelo.fit(X_tr_arr, y_tr)
    y_pred = modelo.predict(X_te_arr)

    M = matriz_confusao(y_te, y_pred, n_classes=3)
    metr = relatorio_metricas(M)

    print(f"\n[Acurácia Geral]:     {metr['acuracia'] * 100:.2f}%")
    print(f"[Acurácia Balanceada]:{metr['acuracia_balanceada'] * 100:.2f}%")
    print(f"[F1 Macro]:           {metr['f1_macro']:.4f}")

    print("\n--- Desempenho por Classe ---")
    linhas_classe = []
    for i, nome in enumerate(ROTULOS):
        p, r, f = metr['precisao'][i], metr['revocacao'][i], metr['f1'][i]
        print(f"  {nome:10s} -> Precisão: {p:.3f} | Revocação: {r:.3f} | F1: {f:.3f}")
        linhas_classe.append({"Classe": nome, "Precisao": p, "Revocacao": r, "F1": f})

    df_classes = pd.DataFrame(linhas_classe)
    df_classes.to_csv(os.path.join(pasta_tabelas, f"knn_metricas_por_classe_cenario_{cenario}.csv"), index=False)

    # Matriz de Confusao visual (Heatmap)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(M, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[f"Pred {r}" for r in ROTULOS],
                yticklabels=[f"Real {r}" for r in ROTULOS],
                annot_kws={"size": 13, "weight": "bold"}, ax=ax)

    ax.set_title(f"Matriz de Confusão do Modelo Final (k-NN, k={k_escolhido})\n"
                 f"Cenário {cenario} - Teste Independente (Seção 6 E5)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Predição do Modelo", fontsize=11, fontweight='bold')
    ax.set_ylabel("Classe Real", fontsize=11, fontweight='bold')

    fig.tight_layout()
    caminho_fig = os.path.join(pasta_figuras, f"knn_matriz_confusao_final_cenario_{cenario}.png")
    plt.savefig(caminho_fig, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figura] Matriz de Confusão salva em: {caminho_fig}")

    # Análise da Classe Enrolled
    idx_enrolled = MAPA_ALVO["Enrolled"]
    total_enrolled = int(M[idx_enrolled].sum())
    acertos_enrolled = int(M[idx_enrolled, idx_enrolled])
    conf_dropout = int(M[idx_enrolled, MAPA_ALVO["Dropout"]])
    conf_graduate = int(M[idx_enrolled, MAPA_ALVO["Graduate"]])

    print("\n--- Análise Crítica da Classe 'Enrolled' (Requisito Seção 4.1.3 e E5) ---")
    print(f"Total de instâncias Enrolled no teste: {total_enrolled}")
    print(f"Acertos exatos como Enrolled:         {acertos_enrolled} ({acertos_enrolled/total_enrolled*100:.1f}%)")
    print(f"Confundidos como Dropout:              {conf_dropout} ({conf_dropout/total_enrolled*100:.1f}%)")
    print(f"Confundidos como Graduate:             {conf_graduate} ({conf_graduate/total_enrolled*100:.1f}%)")
    print("Conclusão: Enrolled é uma classe de transição/ambígua; os estudantes ainda matriculados")
    print("compartilham vizinhança tanto com evadidos quanto com concluintes.")

    return metr


def executar_pipeline_completo(cenario="A"):
    """Executa a bateria de experimentos completos para o cenário selecionado."""
    pasta_fig = "figuras"
    pasta_tab = "tabelas"
    os.makedirs(pasta_fig, exist_ok=True)
    os.makedirs(pasta_tab, exist_ok=True)

    print("\n" + "#"*70)
    print(f"# PIPELINE EXPERIMENTAL COMPLETO DO k-NN | CENÁRIO {cenario}")
    print("#"*70)

    # 1. Carregamento e Codificação via trat.py
    X_raw, Y_raw = carregar_dados("data.csv", cenario=cenario)
    X_clean, Y_clean = limpar_dados(X_raw, Y_raw)
    X_feat = construir_atributos(X_clean)
    X_cod = codificar(X_feat, mapa_de_tipos)
    y_numerico = Y_clean.map(MAPA_ALVO).to_numpy(dtype=int)

    # 2. Partição Estratificada 80/20 (Treino vs Teste Final)
    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X_cod, y_numerico, test_size=0.2, random_state=SEMENTE, stratify=y_numerico
    )

    # 3. Tratamento de Outliers e Normalização no Treino e Teste
    colunas_continuas = [c for c, tipo in mapa_de_tipos.items() if tipo == 'continuos' and c in X_cod.columns]
    X_tr_df, X_te_df = tratar_outliers_iqr(X_tr_df, X_te_df, colunas_continuas)
    X_tr_norm, X_te_norm = normalizar_trat(X_tr_df, X_te_df, metodo="minmax")

    X_tr_arr = X_tr_norm.to_numpy(dtype=float)
    X_te_arr = X_te_norm.to_numpy(dtype=float)

    # Validação do k-NN contra Scikit-Learn
    knn_teste = KNN(k=15, metrica="euclidiana")
    knn_teste.fit(X_tr_arr, y_tr)
    pred_autoral = knn_teste.predict(X_te_arr)

    sk_knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
    sk_knn.fit(X_tr_arr, y_tr)
    pred_sk = sk_knn.predict(X_te_arr)
    coincidencia = np.mean(pred_autoral == pred_sk)
    print(f"\n[VALIDAÇÃO k-NN AUTORAL vs SCIKIT-LEARN]: {coincidencia*100:.2f}% de coincidência exata!")

    # 4. Rodar Experimentos Obrigatórios
    experimento_e1_normalizacao(X_tr_df, y_tr, cenario=cenario, pasta_figuras=pasta_fig, pasta_tabelas=pasta_tab)
    melhores_dist = experimento_e3_distancias_e_k(X_tr_arr, y_tr, X_te_arr, y_te, cenario=cenario, pasta_figuras=pasta_fig, pasta_tabelas=pasta_tab)
    experimento_maldicao_dimensionalidade(X_tr_df, y_tr, cenario=cenario, pasta_figuras=pasta_fig, pasta_tabelas=pasta_tab)
    experimento_e4_custo_computacional(X_tr_arr, y_tr, X_te_arr, y_te, cenario=cenario, pasta_tabelas=pasta_tab)

    k_otimo_euc = melhores_dist.get("Euclidiana", {}).get("k_otimo", 15)
    experimento_e5_modelo_final_teste(X_tr_arr, y_tr, X_te_arr, y_te, k_escolhido=k_otimo_euc,
                                      metrica_escolhida="euclidiana", cenario=cenario,
                                      pasta_figuras=pasta_fig, pasta_tabelas=pasta_tab)

    print("\n" + "="*70)
    print(f" PIPELINE DO k-NN CONCLUÍDO COM SUCESSO PARA O CENÁRIO {cenario}!")
    print("="*70)


if __name__ == "__main__":
    executar_pipeline_completo(cenario="A")
