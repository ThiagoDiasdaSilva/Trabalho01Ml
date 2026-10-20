"""
Script Principal do Projeto Pratico 01 (PP01) - Aprendizado de Maquina
Executa o pipeline completo do dado bruto a recomendacao final
em um unico comando, conforme a Secao 5 do PDF do projeto.
"""
import sys
import time

from trat import carregar_dados, limpar_dados, construir_atributos, codificar, mapa_de_tipos
from meupca import gerar_graficos_pca
from meuknn import executar_pipeline_completo
from minhaarvore import rodar_cenario as rodar_arvore


def main():
    inicio_total = time.perf_counter()
    print("=" * 80)
    print(" RUS0310 - APRENDIZADO DE MAQUINA | PROJETO PRATICO 01 (PP01)")
    print(" Execucao Unificada do Pipeline Completo (Dados -> PCA -> k-NN -> Experimentos)")
    print("=" * 80)

    # 1. Validacao dos dados de entrada
    print("\n>>> ETAPA 1: Carregamento e Preparacao dos Dados")
    X, y = carregar_dados("data.csv", "A")
    X, y = limpar_dados(X, y)
    X = construir_atributos(X)
    X = codificar(X, mapa_de_tipos)
    print(f"Base carregada e codificada com sucesso: {X.shape[0]} amostras e {X.shape[1]} atributos.")

    # 2. Extracao de Caracteristicas por PCA (Unidade 2 e Experimento E2)
    print("\n>>> ETAPA 2: Extracao de Caracteristicas por PCA")
    for cenario in ["A", "B", "C"]:
        gerar_graficos_pca(cenario=cenario)

    # 3. Classificacao e Experimentos do k-NN (Unidade 4 e Experimentos E1 a E5)
    print("\n>>> ETAPA 3: Experimentos do Classificador k-NN")
    for cenario in ["A", "B", "C"]:
        executar_pipeline_completo(cenario=cenario)

    # 4. Arvore de Decisao (Unidade 3 e Experimentos E1, E4 e E5)
    print("\n>>> ETAPA 4: Experimentos da Arvore de Decisao")
    for cenario in ["A", "B", "C"]:
        rodar_arvore(cenario)

    tempo_total = time.perf_counter() - inicio_total
    print("\n" + "=" * 80)
    print(f" PIPELINE COMPLETO CONCLUIDO COM SUCESSO EM {tempo_total:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()