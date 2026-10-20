# PP01 — Predição de Evasão Estudantil no Ensino Superior

**Disciplina:** RUS0310 — Aprendizado de Máquina (2026.2)
**Instituição:** Universidade Federal do Ceará, *Campus* de Russas
**Professor:** Alan Rocha

**Equipe:**

| Integrante | Responsabilidade |
|---|---|
| _Nome 1_ | _ex.: árvore de decisão (Unidade 3)_ |
| _Nome 2_ | _ex.: k-NN (Unidade 4)_ |
| _Nome 3_ | _ex.: PCA e seleção de características (Unidade 2)_ |
| _Nome 4_ | _ex.: pré-processamento e análise de viés_ |

---

## 1. Sobre o projeto

Uma instituição de ensino superior quer **identificar precocemente estudantes em risco de evasão**. O projeto constrói, do dado bruto à recomendação final, um classificador que prevê o desfecho do aluno em três classes:

- `Dropout` (evadido)
- `Enrolled` (ainda matriculado, sem desfecho)
- `Graduate` (concluinte)

O foco não é maximizar a acurácia, e sim responder, com evidência experimental: (i) com que antecedência é possível prever a evasão sem usar informação do futuro; (ii) quais características realmente carregam sinal; (iii) qual modelo entregar à coordenação, considerando desempenho, interpretabilidade e custo.

### Cenários de predição

Para evitar vazamento temporal (*data leakage*), todos os experimentos são executados em três recortes de atributos:

| Cenário | Momento da predição | Atributos incluídos |
|---|---|---|
| **A** | No ato da matrícula | demográficos, socioeconômicos, ingresso, macroeconômicos |
| **B** | Fim do 1º semestre | cenário A + desempenho do 1º semestre |
| **C** | Fim do 1º ano | cenário B + desempenho do 2º semestre |

### Algoritmos (implementação própria)

| Módulo | Conteúdo |
|---|---|
| PCA | extração de características a partir da matriz de covariância, validada contra o scikit-learn |
| k-NN | voto majoritário e ponderado, várias medidas de distância |
| Árvore de decisão | entropia, Gini, ganho de informação, indução ID3/CART, poda, regras SE-ENTÃO e importância dos atributos |

O scikit-learn é usado para partição dos dados, validação cruzada, métricas e como referência de validação. Na busca de hiperparâmetros da árvore, o `DecisionTreeClassifier` é usado por velocidade (permitido pela Seção 4.3 do enunciado). Não são usados AutoML nem os modelos reservados ao PP02 (Naive Bayes, redes neurais, SVM e *ensembles*).

---

## 2. Estrutura do repositório

```
.
├── main.py            # pipeline completo, executado com um único comando
├── trat.py            # carga, limpeza, atributos derivados, codificação e normalização
├── meupca.py          # PCA própria e gráficos de PCA
├── meuknn.py          # k-NN próprio, validação cruzada sem vazamento e experimentos
├── minhaarvore.py     # árvore de decisão própria e experimentos da árvore
├── trabalho1ml.py     # <!-- confirmar a descrição deste arquivo -->
├── requirements.txt   # dependências
├── tabelas/           # resultados em .csv (gerados pelo pipeline)
├── figuras/           # gráficos em .png (gerados pelo pipeline)
└── data.csv           # base de dados (NÃO versionada; veja a seção 3)
```

---

## 3. Como obter a base de dados

A base **não está versionada** neste repositório.

1. Baixe o dataset **Predict Students' Dropout and Academic Success** (UCI Machine Learning Repository, dataset 697):
   <https://archive.ics.uci.edu/dataset/697>
   DOI: `10.24432/C5MC89`
2. Extraia o arquivo CSV do `.zip` baixado.
3. Salve-o como **`data.csv`** na **raiz do projeto** (mesma pasta do `main.py`). O arquivo da UCI usa `;` como separador.

A base tem 4.424 alunos e 36 atributos preditivos, e a variável alvo é `Target`.

---

## 4. Instalação

Requer **Python 3.10 ou superior**.

```bash
# 1. (opcional, recomendado) criar e ativar um ambiente virtual
python -m venv .env

# Windows (PowerShell)
.env\Scripts\Activate.ps1
# Linux / macOS
source .env/bin/activate

# 2. instalar as dependências
pip install -r requirements.txt
```

---

## 5. Como executar

### Pipeline completo (um único comando)

```bash
python main.py
```

Executa, nesta ordem, para os cenários A, B e C:

1. carga e preparação dos dados;
2. extração de características por PCA;
3. experimentos do k-NN;
4. experimentos da árvore de decisão.

O pipeline completo usa validação cruzada 10-fold × 3 repetições e pode levar vários minutos.

### Execuções parciais

```bash
python minhaarvore.py          # roda apenas os testes da árvore (não precisa do data.csv)
python minhaarvore.py A        # experimentos da árvore apenas no cenário A
python minhaarvore.py A B C    # experimentos da árvore nos três cenários
```

Os testes da árvore verificam: o cálculo à mão de entropia, Gini e ganho; a construção da árvore em um exemplo pequeno; o sobreajuste sem limites de profundidade; as regras e a importância dos atributos; a poda; e a proximidade com o `DecisionTreeClassifier`. Eles também geram `figuras/arvore_exemplo_8_alunos.png`.

---

## 6. Resultados gerados

Ao final da execução, os artefatos ficam em:

- **`tabelas/`**: tabelas `.csv` dos experimentos (a árvore salva arquivos com prefixo `arvore_`, por exemplo `arvore_curva_profundidade_cenario_A.csv`, `arvore_regras_cenario_B.csv` e `arvore_importancia_cenario_C.csv`);
- **`figuras/`**: gráficos `.png` (curvas de profundidade e de `k`, projeções de PCA, matrizes de confusão, desenho da árvore final, entre outros).

Os experimentos cobertos são os da Seção 6 do enunciado: E1 (normalização), E2 (PCA), E3 (medidas de distância), E4 (custo computacional) e E5 (modelo final no teste). As análises e a discussão estão no relatório técnico.

---

## 7. Protocolo experimental

| Item | Configuração |
|---|---|
| Semente aleatória | `42` |
| Divisão dos dados | treino / validação / teste = 64% / 16% / 20%, estratificada pela classe (2.831 / 708 / 885 alunos) |
| Seleção de modelo | validação cruzada estratificada, 10-fold, repetida 3 vezes, sobre o treino; resultados como média ± desvio padrão |
| Vazamento | normalização, codificação e tratamento de *outliers* ajustados **dentro de cada fold**, nunca antes da partição |
| Teste | usado **uma única vez**, na avaliação do modelo final (E5) |
| Métrica de decisão | F1 macro |
| Métricas reportadas | acurácia, precisão, revocação e F1 por classe e em média macro, e matriz de confusão |

---

## 8. Referências

- REALINHO, V.; MACHADO, J.; BAPTISTA, L.; MARTINS, M. V. Predicting student dropout and academic success. *Data*, v. 7, n. 11, p. 146, 2022.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning*. 2. ed. Springer, 2008.
- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley, 2000.
- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006.
