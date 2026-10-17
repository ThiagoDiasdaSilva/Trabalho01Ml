import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
import matplotlib
import matplotlib.pyplot as plt

# Agrupamento funcional dos atributos preditivos conforme a Tabela 2 e Tabela 3 da especificacao:
# Demograficos (6 atributos)
grupo_demografico = [
    'Marital status', 'Nacionality', 'Displaced', 'Gender', 'Age at enrollment', 'International'
]

# Socioeconomicos (4 familia + 4 aluno = 8 atributos)
grupo_socioeconomico = [
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Scholarship holder'
]

# Ingresso (7 atributos)
grupo_ingresso = [
    'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Admission grade'
]

# Macroeconomicos (3 atributos)
grupo_macroeconomico = [
    'Unemployment rate', 'Inflation rate', 'GDP'
]

# Desempenho 1o semestre (6 atributos)
grupo_desempenho_1sem = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)'
]

# Desempenho 2o semestre (6 atributos)
grupo_desempenho_2sem = [
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)'
]

COLUNA_ALVO = 'Target'

mapa_de_tipos = {
    'Marital status': 'nominal',
    'Nacionality': 'binary', # agrupado em 1 (portuguesa) e 2 (outras)
    'Displaced': 'binary',
    'Gender': 'binary',
    'Age at enrollment': 'continuos',
    'International': 'binary',
    "Mother's qualification": 'nominal',
    "Father's qualification": 'nominal',
    "Mother's occupation": 'nominal',
    "Father's occupation": 'nominal',
    'Educational special needs': 'binary',
    'Debtor': 'binary',
    'Tuition fees up to date': 'binary',
    'Scholarship holder': 'binary',
    'Application mode': 'nominal',
    'Application order': 'ordinal',
    'Course': 'nominal',
    'Daytime/evening attendance': 'binary',
    'Previous qualification': 'nominal',
    'Previous qualification (grade)': 'continuos',
    'Admission grade': 'continuos',
    'Unemployment rate': 'continuos',
    'Inflation rate': 'continuos',
    'GDP': 'continuos',
}

# --- Dados e pre-processamento ------------------------------------------
def carregar_dados(caminho, cenario):
    """Le o CSV e devolve X, y filtrados pelo cenario ('A', 'B' ou 'C')."""
    if not os.path.exists(caminho):
        alternativas = [
            'data.csv',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data.csv'),
        ]
        for alt in alternativas:
            if os.path.exists(alt):
                caminho = alt
                break

    df = pd.read_csv(caminho, sep=';')
    # Remove espacos e caracteres de escape (como \t) dos nomes das colunas
    df.columns = [c.strip() for c in df.columns]

    if cenario == "A":
        # 6 demograficos + 8 socioeconomicos + 7 ingresso + 3 macroeconomicos = 24
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico 

    elif cenario == "B":
        # Cenario A (24) + 6 desempenho 1o sem = 30
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico + grupo_desempenho_1sem

    elif cenario == "C":
        # Cenario B (30) + 6 desempenho 2o sem = 36
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico + grupo_desempenho_1sem + grupo_desempenho_2sem

    else:
        raise ValueError(f"Cenario invalido: {cenario}. Escolha 'A', 'B' ou 'C'.")

    colunas_existentes = [c for c in colunas if c in df.columns]
    x = df[colunas_existentes].copy()
    y = df[COLUNA_ALVO].copy()

    print(f"\n[carregar_dados] cenario {cenario}: CSV com {df.shape[0]} linhas e {df.shape[1]} colunas")
    print(f"[carregar_dados] X ficou com {x.shape[1]} colunas (esperado: 24 em A, 30 em B, 36 em C) | valores nulos em X: {x.isnull().sum().sum()}")
    print(f"[carregar_dados] distribuicao do alvo: {y.value_counts().to_dict()}")

    return x, y


def construir_atributos(df):
    """Cria os atributos derivados de taxa de aprovacao. Trata divisao por zero."""
    dfA = df.copy()

    def taxa(aprovadas, matriculadas):
        mat = matriculadas.replace(0, 1)
        res = aprovadas / mat
        return np.where(matriculadas == 0, 0.0, res)

    tem_1sem = all(c in dfA.columns for c in ['Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)'])
    tem_2sem = all(c in dfA.columns for c in ['Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (approved)'])
    
    # cenario B tem so o 1o semestre, cenario C tem os dois
    if tem_1sem:
        Enrolled1st = dfA['Curricular units 1st sem (enrolled)']
        Approved1st = dfA['Curricular units 1st sem (approved)']
        dfA['Curricular units 1st sem (aproved) treated'] = taxa(Approved1st, Enrolled1st)
    
    if tem_2sem:
        Enrolled2st = dfA['Curricular units 2nd sem (enrolled)']
        Approved2st = dfA['Curricular units 2nd sem (approved)']
        dfA['Curricular units 2nd sem (aproved) treated'] = taxa(Approved2st, Enrolled2st)
    
    if tem_1sem and tem_2sem:
        TotalEnroled = Enrolled2st + Enrolled1st
        TotalApproved = Approved1st + Approved2st
        dfA['Curricular units total (aproved) treated'] = taxa(TotalApproved, TotalEnroled)
    
    novas = [c for c in dfA.columns if c not in df.columns]
    print(f"\n[construir_atributos] {len(novas)} atributo(s) criado(s): {novas if novas else 'nenhum (cenario sem dados de semestre)'}")
    for c in novas:
        print(f"[construir_atributos]   {c}: min={dfA[c].min():.2f} max={dfA[c].max():.2f} media={dfA[c].mean():.2f}")

    return dfA  


def codificar(X, mapa_de_tipos):
    """One-hot para nominais, agrupamento de raras, passthrough continuos/ordinais/binarios."""
    X = X.copy()
    n_colunas_antes = X.shape[1]

    # Agrupamento de categorias raras da Nacionalidade (1 = Portuguesa, 2 = Demais)
    if 'Nacionality' in X.columns:
        X['Nacionality'] = np.where(X['Nacionality'] == 1, 1, 2) 

    for coluna, tipo in mapa_de_tipos.items():
        col_limpa = coluna.strip()
        nome_real = col_limpa if col_limpa in X.columns else (coluna if coluna in X.columns else None)
        if not nome_real:
            continue

        if tipo == 'nominal':
            dummies = pd.get_dummies(X[nome_real], prefix=nome_real, drop_first=False).astype(int)
            X = X.drop(columns=[nome_real])
            X = pd.concat([X, dummies], axis=1)

        elif isinstance(tipo, list):
            mapa_ordinal = {categoria: i for i, categoria in enumerate(tipo)}
            X[nome_real] = X[nome_real].map(mapa_ordinal)

        elif tipo in ('continuos', 'binary', 'ordinal'):
            pass

    print(f"\n[codificar] colunas antes: {n_colunas_antes} | depois do one-hot: {X.shape[1]}")
    if 'Nacionality' in X.columns:
        print(f"[codificar] Nacionality agrupada em: {X['Nacionality'].value_counts().to_dict()} (1 = portuguesa, 2 = outras)")
    return X


def normalizar(X_treino, X_teste, metodo):
    """Ajusta o escalonador SO no treino e aplica nos dois conjuntos."""
    if metodo == 'zscore':
        media = X_treino.mean(axis=0)
        desvio = X_treino.std(axis=0).replace(0, 1)

        X_treinoScalado = (X_treino - media) / desvio
        X_testeScalado = (X_teste - media) / desvio

    elif metodo == 'minmax':
        minimo = X_treino.min(axis=0)
        maximo = X_treino.max(axis=0)
        faixa = (maximo - minimo).replace(0, 1)

        X_treinoScalado = (X_treino - minimo) / faixa
        X_testeScalado = (X_teste - minimo) / faixa

    elif metodo == 'nenhuma':
        X_treinoScalado = X_treino.copy()
        X_testeScalado = X_teste.copy()

    else:
        raise ValueError(f"Metodo incorreto: {metodo}. Escolha 'minmax', 'zscore' ou 'nenhuma'.")

    X_treinoScalado = pd.DataFrame(X_treinoScalado, columns=X_treino.columns, index=X_treino.index)
    X_testeScalado = pd.DataFrame(X_testeScalado, columns=X_teste.columns, index=X_teste.index)

    print(f"\n[normalizar] metodo: {metodo}")
    print(f"[normalizar] treino: min={X_treinoScalado.min().min():.2f} max={X_treinoScalado.max().max():.2f} media={X_treinoScalado.values.mean():.2f}")
    print(f"[normalizar] teste:  min={X_testeScalado.min().min():.2f} max={X_testeScalado.max().max():.2f} media={X_testeScalado.values.mean():.2f}")

    return X_treinoScalado, X_testeScalado


def limpar_dados(X, Y):
    """Remove possiveis instancias duplicadas."""
    X = X.copy()
    Y = Y.copy()

    duplicata = X.duplicated()
    n_duplicatas = duplicata.sum()

    if n_duplicatas > 0:
        X = X[~duplicata]
        Y = Y[~duplicata]

    print(f"\n[limpar_dados] duplicatas encontradas e removidas: {n_duplicatas} | linhas restantes: {len(X)}")
    return X, Y


def tratar_outliers_iqr(X_treino, X_teste, colunas):
    """Calcula os limites de outliers (metodo IQR) SO no treino e aplica (cap) nos dois conjuntos."""
    X_treino = X_treino.copy()
    X_teste = X_teste.copy()

    for col in colunas:
        if col not in X_treino.columns:
            continue

        Q1 = X_treino[col].quantile(0.25)
        Q3 = X_treino[col].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        cortados_treino = ((X_treino[col] < limite_inferior) | (X_treino[col] > limite_superior)).sum()
        cortados_teste = ((X_teste[col] < limite_inferior) | (X_teste[col] > limite_superior)).sum()
        print(f"[tratar_outliers_iqr]   {col}: [{limite_inferior:.1f}, {limite_superior:.1f}] | ajustados: treino={cortados_treino} teste={cortados_teste}")

        X_treino[col] = X_treino[col].clip(lower=limite_inferior, upper=limite_superior)
        X_teste[col] = X_teste[col].clip(lower=limite_inferior, upper=limite_superior)

    return X_treino, X_teste


if __name__ == "__main__":
    caminho_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv')
    X, Y = carregar_dados(caminho_csv, "A")

    X, Y = limpar_dados(X, Y)
    
    X = construir_atributos(X)
    X = codificar(X, mapa_de_tipos)

    X_treino, X_teste, Y_treino, Y_teste = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print(f"\n[train_test_split] treino: {X_treino.shape[0]} linhas | teste: {X_teste.shape[0]} linhas")
    print(f"[train_test_split] proporcao do alvo no treino: {Y_treino.value_counts(normalize=True).round(3).to_dict()}")
    print(f"[train_test_split] proporcao do alvo no teste:  {Y_teste.value_counts(normalize=True).round(3).to_dict()}")

    colunas_continuas = [c for c, tipo in mapa_de_tipos.items() if tipo == 'continuos' and c in X.columns]
    
    X_treino, X_teste = tratar_outliers_iqr(X_treino, X_teste, colunas_continuas)

    X_treinoScalado, X_testeScalado = normalizar(X_treino, X_teste, "minmax")

    # Exibicao e salvamento dos graficos de boxplot
    fig, axs = plt.subplots(len(colunas_continuas), 1, figsize=(7, 3 * len(colunas_continuas)), dpi=95)
    if len(colunas_continuas) == 1:
        axs = [axs]

    for i, col in enumerate(colunas_continuas):
        axs[i].boxplot(X_treino[col], vert=False)
        axs[i].set_ylabel(col)

    plt.tight_layout()
    pasta_figuras = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figuras')
    os.makedirs(pasta_figuras, exist_ok=True)
    caminho_figura = os.path.join(pasta_figuras, 'boxplots_outliers.png')
    plt.savefig(caminho_figura, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[figura] Grafico de boxplots salvo em: {caminho_figura}")