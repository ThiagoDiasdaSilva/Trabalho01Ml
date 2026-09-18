import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt

grupo_demografico = ['Age at enrollment', 'Gender', 'Nacionality', 'Marital status']
grupo_socioeconomico = ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Tuition fees up to date', 'Debtor']
grupo_ingresso = ['Application mode', 'Application order', 'Course', 'Previous qualification', 'Previous qualification (grade)', 'Admission grade']
grupo_macroeconomico = ['Unemployment rate', 'Inflation rate', 'GDP']
grupo_desempenho_1sem = ['Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)']
grupo_desempenho_2sem = ['Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']

COLUNA_ALVO = 'Target'

mapa_de_tipos = {
    'Application mode': 'nominal',
    'Application order': 'continuos',
    'Course': 'nominal',
    'Daytime/evening attendance\t': 'binary',
    'Previous qualification': 'nominal',
    'Previous qualification (grade)': 'continuos',
    'Nacionality': 'nominal',
    "Mother's qualification": 'nominal',
    "Father's qualification": 'nominal',
    "Mother's occupation": 'nominal',
    "Father's occupation": 'nominal',
    'Admission grade': 'continuos',
    'Displaced': 'binary',
    'Educational special needs': 'binary',
    'Debtor': 'binary',
    'Tuition fees up to date': 'binary',
    'Gender': 'binary',
    'Scholarship holder': 'binary',
    'Age at enrollment': 'continuos',
    'International': 'binary',
    
}

# --- Dados e pre-processamento ------------------------------------------
def carregar_dados(caminho, cenario):

    df=pd.read_csv(caminho, sep=';')

    if(cenario == "A"):
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico 

    elif(cenario == "B"):
        colunas = grupo_desempenho_1sem + grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico 

    elif(cenario == "C"):
        colunas = grupo_desempenho_1sem + grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico + grupo_desempenho_2sem

    x = df[colunas].copy()
    y = df[COLUNA_ALVO].copy()

    return x, y
    
#"""Le o CSV e devolve X, y filtrados pelo cenario (’A’, ’B’ ou ’C’)."""




def construir_atributos(df):
    #"""Cria os atributos derivados. Tratar divisao por zero."""
    
    dfA = df.copy()

    colunas_desempenho = [
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (approved)',
    ]

    if not all(c in dfA.columns for c in colunas_desempenho):
        return dfA
    
    Enrolled1st = dfA['Curricular units 1st sem (enrolled)']
    Approved1st = dfA['Curricular units 1st sem (approved)']
    Enrolled2st = dfA['Curricular units 2nd sem (enrolled)']
    Approved2st = dfA['Curricular units 2nd sem (approved)']

    dfA['Curricular units 1st sem (aproved) treated'] = np.where(
        Enrolled1st == 0, 0, Approved1st / Enrolled1st
    )

    dfA['Curricular units 2nd sem (aproved) treated'] = np.where(
            Enrolled2st == 0, 0, Approved2st / Enrolled2st
        )

    TotalEnroled = Enrolled2st + Enrolled1st
    TotalApproved = Approved1st + Approved2st

    dfA['Curricular units total (aproved) treated'] = np.where(
        TotalEnroled == 0, 0, TotalApproved / TotalEnroled
        )

    return dfA  

    

def codificar(X, mapa_de_tipos):
    #"""One-hot para nominais, ordinal para ordinais, passthrough contínuos."""

    X = X.copy()

    if 'Nacionality' in X.columns:
        X['Nacionality'] = np.where(X['Nacionality'] == 1, X['Nacionality'], 2) 

    for coluna, tipo in mapa_de_tipos.items():
        if coluna not in X.columns:
            continue

        if tipo == 'nominal':
            dummies = pd.get_dummies(X[coluna], prefix=coluna, drop_first=False).astype(int)
            X = X.drop(columns=[coluna])
            X = pd.concat([X, dummies], axis=1)

        elif isinstance(tipo, list):
            mapa_ordinal = {categoria: i for i, categoria in enumerate(tipo)}
            X[coluna] = X[coluna].map(mapa_ordinal)

        elif tipo in ('continuos', 'binary'):
            pass

    return X

        


def normalizar(X_treino, X_teste, metodo):
#"""Ajusta o escalonador SO no treino e aplica nos dois conjuntos."""
 
    if metodo == 'zscore':
        media = X_treino.mean(axis=0)
        desvio = X_treino.std(axis=0)
        desvio = desvio.replace(0, 1)

        X_treinoScalado = (X_treino - media)/(desvio)
        X_testeScalado = (X_teste - media)/ (desvio)

    elif metodo == 'minmax':
        minimo = X_treino.min(axis=0)
        maximo = X_treino.max(axis = 0)
        faixa = np.where(maximo - minimo == 0, 1, maximo - minimo)

        X_treinoScalado = (X_treino - minimo)/(faixa)
        X_testeScalado = (X_teste - minimo)/(faixa)
    else:
        raise ValueError("Metodo incorreto")

    
    X_treinoScalado = pd.DataFrame(
        X_treinoScalado,
        columns=X_treino.columns,
        index=X_treino.index
    )
    X_testeScalado = pd.DataFrame(
        X_testeScalado,
        columns=X_teste.columns,
        index=X_teste.index
    )

    return X_treinoScalado, X_testeScalado

def limpar_dados(X, Y):
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

    X, Y = carregar_dados('/home/thiago/Downloads/PP01/dataset/data.csv', "A")

    X, Y = limpar_dados(X, Y)
    
    X = construir_atributos(X)
    X = codificar(X, mapa_de_tipos)

    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

    colunas_continuas = [c for c, tipo in mapa_de_tipos.items() if tipo == 'continuos' and c in X.columns]

    X_treino, X_teste = tratar_outliers_iqr(X_treino, X_teste, colunas_continuas)

    X_treinoScalado, Y_testeScalado = normalizar(X_treino, X_teste, "minmax")

    #referente a parte de exibição dos graficos

    fig, axs = plt.subplots(len(colunas_continuas), 1, figsize=(7, 3 * len(colunas_continuas)), dpi=95)
    if len(colunas_continuas) == 1:
        axs = [axs]  # 

    for i, col in enumerate(colunas_continuas):
        axs[i].boxplot(X_treino[col], vert=False)
        axs[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()