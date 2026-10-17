import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Importa as funções que você definiu no seu trat.py
from trat import (
    carregar_dados,
    limpar_dados,
    construir_atributos,
    codificar,
    tratar_outliers_iqr,
    normalizar,
    mapa_de_tipos
)

SEMENTE = 42
np.random.seed(SEMENTE)

# Mapeamento do alvo para valores numéricos
MAPA_ALVO = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
ROTULOS = ["Dropout", "Enrolled", "Graduate"]


# =========================================================================
# 1. CLASSE k-NN AUTORAL (Atende à Seção 4.4 do PDF)
# =========================================================================
class KNN:
    """
    Classificador k-NN implementado do zero com suporte a:
    - Voto majoritário simples
    - Voto ponderado pelo inverso da distância (1 / (d + eps))
    - Métricas de distância: Euclidiana e Manhattan
    - Otimização matricial por lotes para bases médias/grandes
    """
    def __init__(self, k=5, metrica="euclidiana", ponderado=False, p=3):
        self.k = k
        self.metrica = metrica
        self.ponderado = ponderado
        self.p = p

    def fit(self, X, y):
        self.X_treino = np.asarray(X, dtype=float)
        self.y_treino = np.asarray(y, dtype=int)
        self.classes = np.unique(self.y_treino)
        return self

    def _distancias_em_lote(self, Q):
        """Calcula matriz de distâncias entre o lote de consulta Q e X_treino."""
        if self.metrica == "euclidiana":
            # Otimizacao via algebra linear: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
            q_norm = np.sum(Q**2, axis=1, keepdims=True)
            tr_norm = np.sum(self.X_treino**2, axis=1, keepdims=True).T
            d2 = np.maximum(q_norm + tr_norm - 2.0 * np.dot(Q, self.X_treino.T), 0.0)
            return np.sqrt(d2)

        elif self.metrica == "manhattan":
            return np.sum(np.abs(Q[:, np.newaxis, :] - self.X_treino[np.newaxis, :, :]), axis=2)

        elif self.metrica == "chebyshev":
            return np.max(np.abs(Q[:, np.newaxis, :] - self.X_treino[np.newaxis, :, :]), axis=2)

        elif self.metrica == "minkowski":
            return np.sum(np.abs(Q[:, np.newaxis, :] - self.X_treino[np.newaxis, :, :]) ** self.p, axis=2) ** (1.0 / self.p)

        elif self.metrica == "cosseno":
            q_norm = np.linalg.norm(Q, axis=1, keepdims=True)
            q_norm = np.where(q_norm == 0, 1.0, q_norm)
            tr_norm = np.linalg.norm(self.X_treino, axis=1, keepdims=True).T
            tr_norm = np.where(tr_norm == 0, 1.0, tr_norm)
            sim = np.dot(Q, self.X_treino.T) / (q_norm * tr_norm)
            return 1.0 - sim

        else:
            raise ValueError(f"Métrica desconhecida: {self.metrica}")

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


# =========================================================================
# 2. FUNÇÕES DE AVALIAÇÃO E MÉTRICAS (Seção 4.5 do PDF)
# =========================================================================
def matriz_confusao(y_real, y_pred, n_classes=3):
    M = np.zeros((n_classes, n_classes), dtype=int)
    for real, pred in zip(y_real, y_pred):
        M[real, pred] += 1
    return M

def relatorio_metricas(M):
    total = M.sum()
    acuracia = np.trace(M) / total

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
        "acuracia": acuracia,
        "acuracia_balanceada": float(np.mean(revocacao)),
        "precisao": np.array(precisao),
        "revocacao": np.array(revocacao),
        "f1": np.array(f1),
        "f1_macro": float(np.mean(f1))
    }


# =========================================================================
# 3. PIPELINE DE EXECUÇÃO USANDO O TRATAMENTO DO TRAT.PY
# =========================================================================
def preparar_dados_com_trat(caminho_csv="data.csv", cenario="A"):
    """Carrega, limpa e codifica os dados usando o seu trat.py."""
    X, Y = carregar_dados(caminho_csv, cenario)
    X, Y = limpar_dados(X, Y)
    X = construir_atributos(X)
    X = codificar(X, mapa_de_tipos)
    return X, Y

def executar_knn_pp01(cenario="A", k=15, metrica="euclidiana", ponderado=False, metodo_norm="minmax"):
    print("=" * 70)
    print(f" PIPELINE k-NN | Cenário: {cenario} | Normalização: {metodo_norm} | k: {k}")
    print("=" * 70)

    # 1. Carregamento e Codificação via trat.py
    X, Y = preparar_dados_com_trat("data.csv", cenario=cenario)
    y_numerico = Y.map(MAPA_ALVO).to_numpy(dtype=int)

    print(f"Instâncias após limpeza: {X.shape[0]}")
    print(f"Atributos após One-Hot Encoding: {X.shape[1]}")

    # 2. Divisão Treino e Teste (80/20) estratificada
    from sklearn.model_selection import train_test_split
    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X, y_numerico, test_size=0.2, random_state=SEMENTE, stratify=y_numerico
    )

    # 3. Tratamento de Outliers (IQR) e Normalização (sem vazamento)
    colunas_continuas = [c for c, tipo in mapa_de_tipos.items() if tipo == 'continuos' and c in X.columns]
    X_tr_df, X_te_df = tratar_outliers_iqr(X_tr_df, X_te_df, colunas_continuas)
    X_tr_scaled, X_te_scaled = normalizar(X_tr_df, X_te_df, metodo=metodo_norm)

    # Converte para array Numpy
    X_tr = X_tr_scaled.to_numpy(dtype=float)
    X_te = X_te_scaled.to_numpy(dtype=float)

    # 4. Treinamento e Teste do Modelo Autoral
    modelo = KNN(k=k, metrica=metrica, ponderado=ponderado)

    t0 = time.perf_counter()
    modelo.fit(X_tr, y_tr)
    tempo_treino = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = modelo.predict(X_te)
    tempo_predicao = time.perf_counter() - t0

    # 5. Métricas de Desempenho
    M = matriz_confusao(y_te, y_pred, n_classes=len(ROTULOS))
    metricas = relatorio_metricas(M)

    print(f"\n[Tempo de Treino]:   {tempo_treino * 1000:.2f} ms")
    print(f"[Tempo de Predição]: {tempo_predicao * 1000:.2f} ms ({len(y_te)} amostras)")
    print(f"\n[Acurácia Geral]:     {metricas['acuracia'] * 100:.2f}%")
    print(f"[Acurácia Balanceada]:{metricas['acuracia_balanceada'] * 100:.2f}%")
    print(f"[F1 Macro]:           {metricas['f1_macro']:.4f}")

    print("\n--- Desempenho por Classe ---")
    for i, nome in enumerate(ROTULOS):
        print(f"  {nome:10s} -> Precisão: {metricas['precisao'][i]:.3f} | Revocação: {metricas['revocacao'][i]:.3f} | F1: {metricas['f1'][i]:.3f}")

    print("\n--- Matriz de Confusão ---")
    df_conf = pd.DataFrame(M, index=[f"Real_{r}" for r in ROTULOS], columns=[f"Pred_{r}" for r in ROTULOS])
    print(df_conf)

    # 6. Validação contra Scikit-Learn (Requisito Seção 4.4 do PDF)
    sk_metrica = "manhattan" if metrica == "manhattan" else "euclidean"
    sk_pesos = "distance" if ponderado else "uniform"
    sk_modelo = KNeighborsClassifier(n_neighbors=k, metric=sk_metrica, weights=sk_pesos)
    sk_modelo.fit(X_tr, y_tr)
    y_pred_sk = sk_modelo.predict(X_te)
    acordo = np.mean(y_pred == y_pred_sk)
    print(f"\n[Validação Scikit-Learn]: {acordo * 100:.2f}% de coincidência exata com o KNeighborsClassifier.")

    return metricas


if __name__ == "__main__":
    # Teste no Cenário A com Min-Max
    executar_knn_pp01(cenario="A", k=15, metrica="euclidiana", ponderado=False, metodo_norm="minmax")
