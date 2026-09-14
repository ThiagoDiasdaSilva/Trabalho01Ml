import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')  
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

    """https://www.youtube.com/@thiagovariavel4502
    pode tirar maritinal status
    """
    """"
    Dados nominais: aplicattion mode, course previous quilification, mother qualification, father qualification
    mother occupation, father occupation, Nacionality
    Passthrough: Displaced, Educational speacial needs, debtor, tuition fees up to date, gender, scholar holder, internacional
    ,daytime
    Continuos: Age at enrollment, Admission grade, application order
    , Previous qualification (grade), Unemployment rate, Inflation rate, GDP
    """

    X = X.copy()

    if 'Nacionality' in X.columns:
        X['Nacionality'] = np.where(X['Nacionality'] == 1, X['Nacionality'], 2) 

    for coluna, tipo in mapa_de_tipos.items():
        if coluna not in X.columns:
            continue

        if tipo == 'nominal':
            dummies = pd.get_dummies(X[coluna], prefix=coluna, drop_first=False)
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
        scaler = StandardScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Metodo incorreto")
        

    scaler.fit(X_treino)
    
    

    X_treinoScalado = pd.DataFrame(
        scaler.transform(X_treino),
        columns=X_treino.columns,
        index=X_treino.index
    )
    X_testeScalado = pd.DataFrame(
        scaler.transform(X_teste),
        columns=X_teste.columns,
        index=X_teste.index
    )

    return X_treinoScalado, X_testeScalado


def remover_colunas(X, colunas_a_remover):
    """Remove colunas indesejadas do conjunto de atributos.

    colunas_a_remover: lista de nomes de coluna a excluir de X.
    """
    return X.drop(columns=colunas_a_remover, errors='ignore')


if __name__ == "__main__":

    X, Y = carregar_dados('/home/thiago/Downloads/PP01/dataset/data.csv', "A")
    X.head()

    X.info()
    print(X.isnull().sum())
    print(X.describe())

    print(X.boxplot())

    fig, axs = plt.subplots(len(X.columns), 1, figsize=(7, 18), dpi=95)
    for i, col in enumerate(X.columns):
        axs[i].boxplot(X[col], vert=False)
        axs[i].set_ylabel(col)
    plt.tight_layout()
    plt.show()