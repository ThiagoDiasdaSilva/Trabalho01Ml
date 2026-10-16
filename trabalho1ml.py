import os
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
    'Marital status': 'nominal',
    'Application mode': 'nominal',
    'Application order': 'ordinal', ####
    'Course': 'nominal',
    'Daytime/evening attendance\t': 'binary',
    'Previous qualification': 'nominal',
    'Previous qualification (grade)': 'continuos',
    'Nacionality': 'binary', ####
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
def carregar_dados(caminho, cenario): ### olhar dps

    df=pd.read_csv(caminho, sep=';')

    if(cenario == "A"):
        colunas = grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico 

    elif(cenario == "B"):
        colunas = grupo_desempenho_1sem + grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico 

    elif(cenario == "C"):
        colunas = grupo_desempenho_1sem + grupo_demografico + grupo_socioeconomico + grupo_ingresso + grupo_macroeconomico + grupo_desempenho_2sem


    x = df[colunas].copy()
    y = df[COLUNA_ALVO].copy()

    print(f"\n[carregar_dados] cenario {cenario}: CSV com {df.shape[0]} linhas e {df.shape[1]} colunas")
    print(f"[carregar_dados] X ficou com {x.shape[1]} colunas | valores nulos em X: {x.isnull().sum().sum()}")
    print(f"[carregar_dados] distribuicao do alvo: {y.value_counts().to_dict()}")
    

    return x, y
#"""Le o CSV e devolve X, y filtrados pelo cenario (’A’, ’B’ ou ’C’)."""




def construir_atributos(df):
    #"""Cria os atributos derivados. Tratar divisao por zero."""
    
    dfA = df.copy()

    def taxa(aprovadas, matriculadas): #### ver isso aqui
        return np.where(matriculadas == 0, 0, aprovadas / matriculadas.replace(0, 1))

    tem_1sem = all(c in dfA.columns for c in ['Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)'])
    tem_2sem = all(c in dfA.columns for c in ['Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (approved)'])
    
    # cenario B tem so o 1o semestre, cenario C tem os dois: cada taxa e criada se as colunas dela existirem
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
    #"""One-hot para nominais, ordinal para ordinais, passthrough contínuos."""

    X = X.copy()
    n_colunas_antes = X.shape[1]

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

        elif tipo in ('continuos', 'binary', 'ordinal'):
            pass

    print(f"\n[codificar] colunas antes: {n_colunas_antes} | depois do one-hot: {X.shape[1]}")
    if 'Nacionality' in X.columns:
        print(f"[codificar] Nacionality agrupada em: {X['Nacionality'].value_counts().to_dict()} (1 = portuguesa, 2 = outras)")
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

    print(f"\n[normalizar] metodo: {metodo}")
    print(f"[normalizar] treino: min={X_treinoScalado.min().min():.2f} max={X_treinoScalado.max().max():.2f} media={X_treinoScalado.values.mean():.2f}")
    print(f"[normalizar] teste:  min={X_testeScalado.min().min():.2f} max={X_testeScalado.max().max():.2f} media={X_testeScalado.values.mean():.2f}")

    return X_treinoScalado, X_testeScalado

def limpar_dados(X, Y):
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

    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(f"\n[train_test_split] treino: {X_treino.shape[0]} linhas | teste: {X_teste.shape[0]} linhas")
    print(f"[train_test_split] proporcao do alvo no treino: {Y_treino.value_counts(normalize=True).round(3).to_dict()}")
    print(f"[train_test_split] proporcao do alvo no teste:  {Y_teste.value_counts(normalize=True).round(3).to_dict()}")

    colunas_continuas = [c for c, tipo in mapa_de_tipos.items() if tipo == 'continuos' and c in X.columns]
    
    X_treino, X_teste = tratar_outliers_iqr(X_treino, X_teste, colunas_continuas)

    X_treinoScalado, X_testeScalado = normalizar(X_treino, X_teste, "minmax")

    #referente a parte de exibição dos graficos

    fig, axs = plt.subplots(len(colunas_continuas), 1, figsize=(7, 3 * len(colunas_continuas)), dpi=95)
    if len(colunas_continuas) == 1:
        axs = [axs]  # 

    for i, col in enumerate(colunas_continuas):
        axs[i].boxplot(X_treino[col], vert=False)
        axs[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()