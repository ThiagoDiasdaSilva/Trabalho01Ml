import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2

# Importa o pipeline de tratamento existente
from trat import (
    carregar_dados,
    limpar_dados,
    construir_atributos,
    codificar,
    tratar_outliers_iqr,
    normalizar as normalizar_trat,
    mapa_de_tipos
)


def normalizar(X_treino, X_teste=None, metodo='padronizacao'):
    """
    Ajusta o escalonador SO no treino e aplica nos dois conjuntos.
    metodo: 'padronizacao' (Z-score) ou 'minmax'.
    """
    X_tr = np.array(X_treino, dtype=float)

    if metodo in ('padronizacao', 'zscore'):
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


def pca_ajustar(X_treino, n_componentes=None):
    """
    Centraliza, calcula a matriz de covariancia, decompoe em autovalores/autovetores
    e ordena por autovalor decrescente conforme Seção 4.2.b.1 do PDF.
    Devolve: (media, W, variancia_explicada, autovalores)
    """
    X_treino = np.array(X_treino, dtype=float)

    # Centralizacao subtraindo a media de cada atributo
    media = np.mean(X_treino, axis=0)
    X_cent = X_treino - media

    # Matriz de covariancia amostral
    cov = np.cov(X_cent, rowvar=False)

    # Autovalores e autovetores via decomposicao espectral hermitiana
    autovalores, autovetores = np.linalg.eigh(cov)

    # Ordenacao decrescente
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    autovetores = autovetores[:, idx]

    # Garante nao-negatividade numerica de autovalores
    autovalores = np.maximum(autovalores, 0.0)

    # Determinacao da quantidade de componentes
    if n_componentes is None or n_componentes > X_treino.shape[1]:
        n_componentes = X_treino.shape[1]

    W = autovetores[:, :n_componentes]

    # Convencao de sinal consistente (bater com o scikit-learn)
    for i in range(W.shape[1]):
        max_abs_idx = np.argmax(np.abs(W[:, i]))
        if W[max_abs_idx, i] < 0:
            W[:, i] = -W[:, i]

    # Variancia explicada individual
    variancia_total = np.sum(autovalores)
    if variancia_total == 0:
        variancia_explicada = np.zeros(n_componentes)
    else:
        variancia_explicada = autovalores[:n_componentes] / variancia_total

    return media, W, variancia_explicada


def pca_projetar(X, media, W):
    """
    Projeta as amostras no subespaco de componentes principais:
    Z = (X - media) @ W
    """
    X = np.array(X, dtype=float)
    X_cent = X - media
    Z = X_cent @ W
    return Z


def cargas_componentes(W, nomes_atributos, n=3):
    """
    Calcula as cargas (loadings) das n primeiras componentes principais.
    Permite interpretar a contribuicao de cada atributo conforme Seção 4.2.b do PDF.
    """
    n_reais = min(n, W.shape[1])
    cols = [f"PC{i+1}" for i in range(n_reais)]
    df = pd.DataFrame(W[:, :n_reais], index=nomes_atributos, columns=cols)
    return df


def validar_contra_sklearn(X_treino, n_componentes=5):
    """
    Valida a implementacao autoral de PCA contra o PCA do scikit-learn (Seção 4.2.b.1).
    """
    print()
    print(f"[VALIDAÇÃO PCA AUTORAL vs SCIKIT-LEARN (n_componentes = {n_componentes})]")
    print()

    # PCA Autoral
    media, W, var_exp_autoral = pca_ajustar(X_treino, n_componentes=n_componentes)
    Z_autoral = pca_projetar(X_treino, media, W)

    # PCA do Scikit-Learn
    pca_sk = SklearnPCA(n_components=n_componentes)
    Z_sk = pca_sk.fit_transform(X_treino)
    var_exp_sk = pca_sk.explained_variance_ratio_

    diff_var = np.max(np.abs(var_exp_autoral - var_exp_sk))

    for j in range(min(Z_autoral.shape[1], Z_sk.shape[1])):
        if np.dot(Z_autoral[:, j], Z_sk[:, j]) < 0:
            Z_sk[:, j] *= -1
    diff_Z = np.max(np.abs(Z_autoral - Z_sk))

    print(f"Diferença máxima na Variância Explicada: {diff_var:.2e}")
    print(f"Diferença máxima nas Projeções Z:         {diff_Z:.2e}")

    if diff_var < 1e-10 and diff_Z < 1e-10:
        print(">>> SUCESSO: A implementação autoral coincide exatamente com o Scikit-Learn! <<<")
    else:
        print(">>> AVISO: Pequena divergência numérica ou de sinal observada. <<<")

    return diff_var, diff_Z


def gerar_graficos_pca(cenario="A", pasta_figuras="figuras", pasta_tabelas="tabelas"):
    """
    Executa a analise completa de PCA para o cenario especificado e gera:
    1. Scree Plot (Variancia Explicada Individual e Acumulada com limiares 90% e 95%)
    2. Projecao 2D colorida por classe (comentando separabilidade e Enrolled)
    3. Comparacao lado a lado: PCA COM vs. SEM Padronizacao
    4. Tabela de Cargas (Loadings) das 3 primeiras componentes
    """
    os.makedirs(pasta_figuras, exist_ok=True)
    os.makedirs(pasta_tabelas, exist_ok=True)

    print("\n" + "="*70)
    print(f" [GERANDO ANÁLISES E GRÁFICOS DO PCA | Cenário {cenario}]")
    print("="*70)

    # Carregamento e Preparacao dos Dados
    caminho_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv')
    X_raw, y = carregar_dados(caminho_csv, cenario)
    X_clean, y = limpar_dados(X_raw, y)
    X_feat = construir_atributos(X_clean)
    X_cod = codificar(X_feat, mapa_de_tipos)
    nomes_atributos = list(X_cod.columns)

    # PCA com Padronizacao Z-Score 
    X_padronizado = normalizar(X_cod, metodo='padronizacao')
    media_pad, W_pad, var_exp_pad = pca_ajustar(X_padronizado, n_componentes=X_padronizado.shape[1])
    Z_pad = pca_projetar(X_padronizado, media_pad, W_pad)

    # Validacao contra Scikit-Learn
    validar_contra_sklearn(X_padronizado, n_componentes=5)

    # Variancia Acumulada
    var_acum_pad = np.cumsum(var_exp_pad)
    d_90 = int(np.searchsorted(var_acum_pad, 0.90) + 1)
    d_95 = int(np.searchsorted(var_acum_pad, 0.95) + 1)

    print(f"\nVariância explicada PC1: {var_exp_pad[0]*100:.2f}% | PC2: {var_exp_pad[1]*100:.2f}% | PC3: {var_exp_pad[2]*100:.2f}%")
    print(f"Variância acumulada PC1 + PC2: {var_acum_pad[1]*100:.2f}%")
    print(f"\nComponentes necessárias para 90% da variância: d = {d_90}")
    print(f"Componentes necessárias para 95% da variância: d = {d_95}")

    # Salva variancia explicada em CSV
    df_var = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(len(var_exp_pad))],
        "Variancia_Individual": var_exp_pad,
        "Variancia_Acumulada": var_acum_pad
    })
    caminho_csv_var = os.path.join(pasta_tabelas, f"pca_variancia_cenario_{cenario}.csv")
    df_var.to_csv(caminho_csv_var, index=False)

    # Gráfico 1: Variancia Explicada Individual e Acumulada)
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Mostra ate as primeiras 50 componentes para visualizacao clara do cotovelo
    n_plot = min(50, len(var_exp_pad))
    x_indices = np.arange(1, n_plot + 1)

    cor_barra = "#3A86FF"
    cor_linha = "#E63946"

    # Barras: variancia individual
    ax1.bar(x_indices, var_exp_pad[:n_plot] * 100, color=cor_barra, alpha=0.7, width=0.6, label='Variância Individual (%)')
    ax1.set_xlabel('Número da Componente Principal', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variância Explicada Individual (%)', color=cor_barra, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=cor_barra)
    ax1.set_xlim(0.5, n_plot + 0.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Eixo secundario: variancia acumulada
    ax2 = ax1.twinx()
    ax2.plot(x_indices, var_acum_pad[:n_plot] * 100, color=cor_linha, marker='o', linewidth=2, markersize=4, label='Variância Acumulada (%)')
    ax2.axhline(90, color='#2B9348', linestyle='--', linewidth=1.5, label='Limiar de 90%')
    ax2.axhline(95, color='#F77F00', linestyle=':', linewidth=1.5, label='Limiar de 95%')
    ax2.set_ylabel('Variância Explicada Acumulada (%)', color=cor_linha, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=cor_linha)
    ax2.set_ylim(0, 105)

    # Anotacoes dos limiares
    ax2.scatter([d_90], [var_acum_pad[d_90 - 1] * 100], color='#2B9348', s=80, zorder=5)
    ax2.annotate(f"d = {d_90} (90%)", xy=(d_90, var_acum_pad[d_90 - 1] * 100),
                 xytext=(d_90 - 12, var_acum_pad[d_90 - 1] * 100 - 8),
                 arrowprops=dict(facecolor='#2B9348', shrink=0.08, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', color='#2B9348')

    plt.title(f"Scree Plot - Variância Explicada por Componente Principal (Cenário {cenario})\n"
              f"Critério de Seleção de Dimensão d", fontsize=13, fontweight='bold', pad=15)

    fig.tight_layout()
    caminho_fig1 = os.path.join(pasta_figuras, f"pca_scree_plot_cenario_{cenario}.png")
    plt.savefig(caminho_fig1, bbox_inches='tight')
    plt.close(fig)

    # Gráfico 2: Projecao 2D (PC1 x PC2)
    paleta_classes = {
        'Dropout': '#E63946',   # Vermelho
        'Enrolled': '#F4A261',  # Âmbar / Laranja
        'Graduate': '#2A9D8F'   # Verde azulado
    }

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

    for classe, cor in paleta_classes.items():
        mascara = (y == classe)
        ax.scatter(Z_pad[mascara, 0], Z_pad[mascara, 1],
                   label=classe, color=cor, alpha=0.6, edgecolors='none', s=24)

    ax.set_title(f"Projeção 2D do PCA (PC1 vs PC2) - Cenário {cenario}\n"
                 f"Separação das Classes e Sobreposição de 'Enrolled' ",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel(f"PC1 ({var_exp_pad[0]*100:.2f}% da variância explicada)", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"PC2 ({var_exp_pad[1]*100:.2f}% da variância explicada)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title="Desfecho do Estudante", frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

    # Anotacao explicativa no grafico sobre a sobreposicao de Enrolled
    ax.text(0.03, 0.04,
            "Observação:\n"
            "• Dropout e Graduate apresentam relativa separação ao longo do espaço projetado.\n"
            "• A classe Enrolled (matrícula ativa) situa-se como um estado intermediário\n"
            "  fortemente sobreposto entre ambas as classes terminais.",
            transform=ax.transAxes, fontsize=9.5, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CED4DA', alpha=0.9))

    fig.tight_layout()
    caminho_fig2 = os.path.join(pasta_figuras, f"pca_projecao_2d_cenario_{cenario}.png")
    plt.savefig(caminho_fig2, bbox_inches='tight')
    plt.close(fig)

    # Gráfico 3: COM vs. SEM Padronizacao
    X_sem_pad = np.array(X_cod, dtype=float)
    media_sem, W_sem, var_exp_sem = pca_ajustar(X_sem_pad, n_componentes=2)
    Z_sem = pca_projetar(X_sem_pad, media_sem, W_sem)

    fig, (ax_sem, ax_com) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    # Sem Padronizacao
    for classe, cor in paleta_classes.items():
        mascara = (y == classe)
        ax_sem.scatter(Z_sem[mascara, 0], Z_sem[mascara, 1],
                       label=classe, color=cor, alpha=0.55, edgecolors='none', s=20)
    ax_sem.set_title(f"(A) SEM Padronização Prévia\n"
                     f"PC1 ({var_exp_sem[0]*100:.1f}%) e PC2 ({var_exp_sem[1]*100:.1f}%)\n"
                     f"Dominado por notas/idades de grande escala", fontsize=11, fontweight='bold')
    ax_sem.set_xlabel("PC1 (escala original não normalizada)", fontsize=10, fontweight='bold')
    ax_sem.set_ylabel("PC2 (escala original não normalizada)", fontsize=10, fontweight='bold')
    ax_sem.grid(True, linestyle='--', alpha=0.3)
    ax_sem.legend(frameon=True, loc='upper right')

    # Com Padronizacao Z-score
    for classe, cor in paleta_classes.items():
        mascara = (y == classe)
        ax_com.scatter(Z_pad[mascara, 0], Z_pad[mascara, 1],
                       label=classe, color=cor, alpha=0.55, edgecolors='none', s=20)
    ax_com.set_title(f"COM Padronização (Z-Score)\n"
                     f"PC1 ({var_exp_pad[0]*100:.1f}%) e PC2 ({var_exp_pad[1]*100:.1f}%)\n"
                     f"Todas as variáveis contribuem na mesma escala", fontsize=11, fontweight='bold')
    ax_com.set_xlabel(f"PC1 ({var_exp_pad[0]*100:.1f}% da variância)", fontsize=10, fontweight='bold')
    ax_com.set_ylabel(f"PC2 ({var_exp_pad[1]*100:.1f}% da variância)", fontsize=10, fontweight='bold')
    ax_com.grid(True, linestyle='--', alpha=0.3)
    ax_com.legend(frameon=True, loc='upper right')

    fig.suptitle(f"Efeito Crítico da Padronização Prévia no PCA {cenario}",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    caminho_fig3 = os.path.join(pasta_figuras, f"pca_com_vs_sem_padronizacao_cenario_{cenario}.png")
    plt.savefig(caminho_fig3, bbox_inches='tight')
    plt.close(fig)

   
    # TABELA DE CARGAS (LOADINGS)
    df_cargas = cargas_componentes(W_pad, nomes_atributos, n=3)
    caminho_csv_cargas = os.path.join(pasta_tabelas, f"pca_cargas_loadings_cenario_{cenario}.csv")
    df_cargas.to_csv(caminho_csv_cargas)

    print("\n[Top 5 Atributos com Maiores Cargas Absolutas em PC1]")
    pc1_top = df_cargas['PC1'].abs().sort_values(ascending=False).head(5)
    for attr in pc1_top.index:
        print(f"  {attr:45s} -> Carga: {df_cargas.loc[attr, 'PC1']:+.4f}")

    print("\n[Top 5 Atributos com Maiores Cargas Absolutas em PC2]")
    pc2_top = df_cargas['PC2'].abs().sort_values(ascending=False).head(5)
    for attr in pc2_top.index:
        print(f"  {attr:45s} -> Carga: {df_cargas.loc[attr, 'PC2']:+.4f}")

    return {
        "var_exp": var_exp_pad,
        "var_acum": var_acum_pad,
        "d_90": d_90,
        "d_95": d_95,
        "cargas": df_cargas
    }


if __name__ == "__main__":
    # Executa a analise de PCA completa para o Cenário A (ou todos)
    res_a = gerar_graficos_pca(cenario="A")
    print("\nExecução do PCA finalizada com sucesso!")