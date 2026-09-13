import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')  # ou 'Qt5Agg', dependendo do que estiver instalado
import matplotlib.pyplot as plt

grupo_demografico = ['Age at enrollment', 'Gender', 'Nacionality', 'Marital status']
grupo_socioeconomico = ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Tuition fees up to date', 'Debtor']
grupo_ingresso = ['Application mode', 'Application order', 'Course', 'Previous qualification', 'Previous qualification (grade)', 'Admission grade']
grupo_macroeconomico = ['Unemployment rate', 'Inflation rate', 'GDP']
grupo_desempenho_1sem = ['Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)']
grupo_desempenho_2sem = ['Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']

COLUNA_ALVO = ['Target']

# --- Dados e pre-processamento ------------------------------------------
def carregar_dados(caminho, cenario):

    df=pd.read_csv(caminho, sep=';')

    if(cenario == "A"):
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico 

    elif(cenario == "B"):
        colunas = grupo_desempenho_1sem + grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico 

    elif(cenario == "C"):
        colunas = grupo_desempenho_1sem + grupo_demografico + grupo_socioeconomico + grupo_ingresso,+ grupo_macroeconomico + grupo_desempenho_2sem

    x = df[colunas].copy()
    y = df[COLUNA_ALVO].copy()

    return x, y
    
#"""Le o CSV e devolve X, y filtrados pelo cenario (’A’, ’B’ ou ’C’)."""




def construir_atributos(df):
    #"""Cria os atributos derivados. Tratar divisao por zero."""

    df['Nacionality'] = np.where(df['Nacionality'] == 1, df['Nacionality'], 2) #tudo que for diferente de 1  classicar como, "outros"
    return df



#def codificar(X, mapa_de_tipos):
    #"""One-hot para nominais, ordinal para ordinais, passthrough contínuos."""




#def normalizar(X_treino, X_teste, metodo):
#"""Ajusta o escalonador SO no treino e aplica nos dois conjuntos."""
 



    # (x - x_min) / (x_max - x_min)

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