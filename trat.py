import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
import matplotlib
import matplotlib.pyplot as plt

# Agrupamento funcional dos atributos preditivos conforme Tabelas 2 e 3 da especificacao
grupo_demografico = [
    'Marital status', 'Nacionality', 'Displaced', 'Gender', 'Age at enrollment', 'International'
]

grupo_socioeconomico = [
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Scholarship holder'
]

grupo_ingresso = [
    'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Admission grade'
]

grupo_macroeconomico = [
    'Unemployment rate', 'Inflation rate', 'GDP'
]

grupo_desempenho_1sem = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)'
]

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
    df.columns = [c.strip() for c in df.columns]

    if cenario == "A":
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico 
    elif cenario == "B":
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico + grupo_desempenho_1sem
    elif cenario == "C":
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico + grupo_desempenho_1sem + grupo_desempenho_2sem
    else:
        raise ValueError(f"Cenario invalido: {cenario}")

    colunas = [c for c in colunas if c in df.columns]
    x = df[colunas].copy()
    y = df[COLUNA_ALVO].copy()

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

    return dfA  

def codificar(X, mapa_de_tipos):
    """One-hot para nominais, agrupamento de raras, passthrough continuos/ordinais/binarios."""
    X = X.copy()

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
        raise ValueError(f"Metodo incorreto: {metodo}")

    X_treinoScalado = pd.DataFrame(X_treinoScalado, columns=X_treino.columns, index=X_treino.index)
    X_testeScalado = pd.DataFrame(X_testeScalado, columns=X_teste.columns, index=X_teste.index)

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

        X_treino[col] = X_treino[col].clip(lower=limite_inferior, upper=limite_superior)
        X_teste[col] = X_teste[col].clip(lower=limite_inferior, upper=limite_superior)

    return X_treino, X_teste


if __name__ == "__main__":
    caminho = 'data.csv' if os.path.exists('data.csv') else os.path.join(os.path.dirname(__file__), 'data.csv')
    X, Y = carregar_dados(caminho, "A")

    X, Y = limpar_dados(X, Y)
    X = construir_atributos(X)
    X = codificar(X, mapa_de_tipos)

    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    colunas_continuas = [c for c, tipo in mapa_de_tipos.items() if tipo == 'continuos' and c in X.columns]
    X_treino, X_teste = tratar_outliers_iqr(X_treino, X_teste, colunas_continuas)
    X_treinoScalado, X_testeScalado = normalizar(X_treino, X_teste, "minmax")

    print(f"Dados preparados com sucesso: Treino {X_treinoScalado.shape}, Teste {X_testeScalado.shape}")