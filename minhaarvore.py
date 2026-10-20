"""
Como rodar:
  python minhaarvore.py        -> roda só os testes (não precisa do data.csv)
  python minhaarvore.py A      -> roda os experimentos da árvore no cenário A
  python minhaarvore.py A B C  -> roda nos três cenários
"""

import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEMENTE = 42
ROTULOS = ["Dropout", "Enrolled", "Graduate"] 
PASTA_FIGURAS = "figuras"
PASTA_TABELAS = "tabelas"


# 1. Entropia, Gini e ganho de informação
def entropia(y):
    _, cont = np.unique(y, return_counts=True)
    p = cont / cont.sum()
    return -np.sum(p * np.log2(p))


def gini(y):
    _, cont = np.unique(y, return_counts=True)
    p = cont / cont.sum()
    return 1.0 - np.sum(p ** 2)


def ganho_informacao(coluna, y, limiar, criterio="entropia"):
    imp = entropia if criterio == "entropia" else gini
    esq = coluna <= limiar
    if esq.all() or (~esq).all():
        return 0.0
    n = len(y)
    return imp(y) - esq.sum() / n * imp(y[esq]) - (~esq).sum() / n * imp(y[~esq])


# 2. Melhor divisão de um nó
def candidatos_limiar(coluna, max_cand=20):
    """Limiares que vale a pena testar: pontos médios entre valores vizinhos.
    Se a coluna tem muitos valores (contínua), usa só alguns percentis
    para não ficar lento (o sklearn testa todos)."""
    valores = np.unique(coluna)
    if len(valores) < 2:
        return []
    if len(valores) > max_cand + 1:
        valores = np.unique(np.percentile(coluna, np.linspace(0, 100, max_cand + 1)))
    return (valores[:-1] + valores[1:]) / 2


def melhor_divisao(X, y, criterio="gini", min_folha=5):
    """Testa cada atributo e cada limiar candidato e devolve (atributo, limiar, ganho)."""
    melhor_atributo, melhor_limiar, melhor_ganho = None, None, 0.0
    for j in range(X.shape[1]):
        for lim in candidatos_limiar(X[:, j]):
            esq = X[:, j] <= lim
            if esq.sum() < min_folha or (~esq).sum() < min_folha:
                continue   # filho pequeno demais
            g = ganho_informacao(X[:, j], y, lim, criterio)
            if g > melhor_ganho:
                melhor_atributo, melhor_limiar, melhor_ganho = j, float(lim), float(g)
    return melhor_atributo, melhor_limiar, melhor_ganho


# 3. A árvore
class No:
    def __init__(self, y, n_classes=3):
        self.contagens = np.bincount(y, minlength=n_classes)   # alunos de cada classe neste nó
        self.classe = int(np.argmax(self.contagens))           # classe majoritária
        self.atributo = None
        self.limiar = None
        self.esq = None
        self.dir = None
        self.ganho = 0.0

    def eh_folha(self):
        return self.esq is None


def induzir_arvore(X, y, criterio="gini", prof_max=5, min_folha=5, prof_atual=0):
    no = No(y)

    # quando parar (vira folha)
    if prof_atual >= prof_max or len(np.unique(y)) == 1 or len(y) < 2 * min_folha:
        return no

    j, lim, ganho = melhor_divisao(X, y, criterio, min_folha)
    if j is None:
        return no

    esq = X[:, j] <= lim
    no.atributo, no.limiar, no.ganho = j, lim, ganho
    no.esq = induzir_arvore(X[esq], y[esq], criterio, prof_max, min_folha, prof_atual + 1)
    no.dir = induzir_arvore(X[~esq], y[~esq], criterio, prof_max, min_folha, prof_atual + 1)
    return no


def prever_um(no, x):
    """Desce pela árvore respondendo as perguntas até chegar numa folha."""
    while not no.eh_folha():
        if x[no.atributo] <= no.limiar:
            no = no.esq
        else:
            no = no.dir
    return no.classe


def contar_nos(no):
    if no.eh_folha():
        return 1
    return 1 + contar_nos(no.esq) + contar_nos(no.dir)


def profundidade(no):
    if no.eh_folha():
        return 0
    return 1 + max(profundidade(no.esq), profundidade(no.dir))


class ArvoreDecisao:
    def __init__(self, criterio="gini", prof_max=5, min_folha=5):
        self.criterio = criterio
        self.prof_max = prof_max
        self.min_folha = min_folha
        self.raiz = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_atributos = X.shape[1]
        self.raiz = induzir_arvore(X, y, self.criterio, self.prof_max, self.min_folha)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([prever_um(self.raiz, x) for x in X], dtype=int)


# 4. Poda, regras e importância dos atributos
def podar(no, X, y):
    """Poda por erro reduzido (de baixo para cima), usando dados de validação.
    Se trocar uma subárvore por uma folha não aumenta os erros, troca."""
    if no.eh_folha():
        return
    esq = X[:, no.atributo] <= no.limiar
    podar(no.esq, X[esq], y[esq])
    podar(no.dir, X[~esq], y[~esq])

    erros_arvore = sum(prever_um(no, x) != rotulo for x, rotulo in zip(X, y))
    erros_folha = np.sum(y != no.classe)
    if erros_folha <= erros_arvore:
        no.esq = None
        no.dir = None
        no.atributo = None
        no.limiar = None


def extrair_regras(no, nomes_atributos, nomes_classes, condicoes=None, n_total=None):
    """Uma regra SE-ENTÃO por folha, com cobertura e confiança."""
    if condicoes is None:
        condicoes = []
    if n_total is None:
        n_total = no.contagens.sum()

    if no.eh_folha():
        n = no.contagens.sum()
        return [{
            "regra": "SE " + " E ".join(condicoes) + " ENTÃO " + nomes_classes[no.classe],
            "cobertura": n / n_total,
            "confianca": no.contagens[no.classe] / n,
        }]

    nome = nomes_atributos[no.atributo]
    regras = extrair_regras(no.esq, nomes_atributos, nomes_classes,
                            condicoes + [f"{nome} <= {no.limiar:.4g}"], n_total)
    regras += extrair_regras(no.dir, nomes_atributos, nomes_classes,
                             condicoes + [f"{nome} > {no.limiar:.4g}"], n_total)
    return regras


def importancia_atributos(raiz, n_atributos):
    """Soma o ganho de cada nó (ponderado pela fração de alunos que passa por ele)."""
    imp = np.zeros(n_atributos)
    n_raiz = raiz.contagens.sum()

    def percorrer(no):
        if no.eh_folha():
            return
        imp[no.atributo] += no.ganho * no.contagens.sum() / n_raiz
        percorrer(no.esq)
        percorrer(no.dir)

    percorrer(raiz)
    return imp / imp.sum() if imp.sum() > 0 else imp


# 5. Métrica simples e curva de profundidade
def f1_macro(y_real, y_pred, n_classes=3):
    y_real = np.asarray(y_real, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    f1s = []
    for c in range(n_classes):
        vp = np.sum((y_real == c) & (y_pred == c))
        fp = np.sum((y_real != c) & (y_pred == c))
        fn = np.sum((y_real == c) & (y_pred != c))
        p = vp / (vp + fp) if vp + fp > 0 else 0.0
        r = vp / (vp + fn) if vp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return float(np.mean(f1s))


def curva_profundidade(X_tr, y_tr, X_val, y_val, profundidades, criterio="gini", min_folha=5):
    linhas = []
    for p in profundidades:
        arv = ArvoreDecisao(criterio, p, min_folha).fit(X_tr, y_tr)
        linhas.append({
            "prof_max": p,
            "f1_treino": f1_macro(y_tr, arv.predict(X_tr)),
            "f1_validacao": f1_macro(y_val, arv.predict(X_val)),
        })
        print(f"  prof_max={p:2d} | F1 treino={linhas[-1]['f1_treino']:.4f} "
              f"| F1 validação={linhas[-1]['f1_validacao']:.4f}")
    return pd.DataFrame(linhas)


def plotar_curva(df, cenario):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    melhor = int(df.loc[df["f1_validacao"].idxmax(), "prof_max"])
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(df["prof_max"], df["f1_treino"] * 100, marker="o", label="Treino")
    plt.plot(df["prof_max"], df["f1_validacao"] * 100, marker="o", label="Validação")
    plt.axvline(melhor, color="gray", linestyle="--", label=f"Melhor validação (prof={melhor})")
    plt.title(f"Árvore de decisão: efeito da profundidade (Cenário {cenario})")
    plt.xlabel("max_depth")
    plt.ylabel("F1 macro (%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIGURAS, f"arvore_curva_profundidade_cenario_{cenario}.png"))
    plt.close()
    return melhor


# 6. Experimentos 
def preparar_dados(cenario):
    """Mesmos passos e mesmos splits do pipeline do k-NN (semente 42)."""
    from sklearn.model_selection import train_test_split
    from trat import carregar_dados, limpar_dados, construir_atributos, codificar, mapa_de_tipos
    from meuknn import MAPA_ALVO

    X, y = carregar_dados("data.csv", cenario)
    X, y = limpar_dados(X, y)
    X = construir_atributos(X)
    X = codificar(X, mapa_de_tipos)
    y = y.map(MAPA_ALVO).to_numpy(dtype=int)

    X_temp, X_te, y_temp, y_te = train_test_split(X, y, test_size=0.2, random_state=SEMENTE, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=SEMENTE, stratify=y_temp)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def comparar_com_sklearn(X_tr, y_tr, X_val, y_val, cenario):
    from sklearn.tree import DecisionTreeClassifier
    print("\n[Árvore própria x scikit-learn]")
    linhas = []
    for prof in [3, 5, 10]:
        minha = ArvoreDecisao("gini", prof, 5).fit(X_tr, y_tr)
        sk = DecisionTreeClassifier(criterion="gini", max_depth=prof, min_samples_leaf=5,
                                    random_state=SEMENTE).fit(X_tr, y_tr)
        f_minha = f1_macro(y_val, minha.predict(X_val))
        f_sk = f1_macro(y_val, sk.predict(X_val))
        linhas.append({"prof_max": prof, "F1_propria": f_minha, "F1_sklearn": f_sk})
        print(f"  prof_max={prof:2d} | F1 própria={f_minha:.4f} | F1 sklearn={f_sk:.4f}")
    pd.DataFrame(linhas).to_csv(os.path.join(PASTA_TABELAS, f"arvore_vs_sklearn_cenario_{cenario}.csv"), index=False)


def escolher_profundidade_cv(X_tr_df, y_tr, n_folds, n_repeticoes):
    """Escolhe prof_max por validação cruzada (usa a árvore do sklearn por ser mais rápida)."""
    from sklearn.tree import DecisionTreeClassifier
    from meuknn import validacao_cruzada_sem_vazamento
    print("\n[Escolha de prof_max por validação cruzada]")
    melhor_prof, melhor_f1 = None, -1
    for prof in [3, 5, 7, 10, 15]:
        res = validacao_cruzada_sem_vazamento(
            X_tr_df, y_tr,
            modelo_builder=lambda p=prof: DecisionTreeClassifier(max_depth=p, min_samples_leaf=5, random_state=SEMENTE),
            n_folds=n_folds, n_repeticoes=n_repeticoes, metodo_norm="minmax")
        print(f"  prof_max={prof:2d} | F1 macro = {res['f1_macro_media']:.4f} ± {res['f1_macro_std']:.4f}")
        if res["f1_macro_media"] > melhor_f1:
            melhor_prof, melhor_f1 = prof, res["f1_macro_media"]
    print(f"  Melhor: prof_max={melhor_prof}")
    return melhor_prof, melhor_f1


def experimento_e1(X_tr_df, y_tr, prof, cenario, n_folds, n_repeticoes):
    """E1 (lado da árvore): com e sem normalização."""
    from sklearn.tree import DecisionTreeClassifier
    from meuknn import validacao_cruzada_sem_vazamento
    print("\n[E1 - Árvore: normalização]")
    linhas = []
    for metodo in ["nenhuma", "minmax", "zscore"]:
        res = validacao_cruzada_sem_vazamento(
            X_tr_df, y_tr,
            modelo_builder=lambda: DecisionTreeClassifier(max_depth=prof, min_samples_leaf=5, random_state=SEMENTE),
            n_folds=n_folds, n_repeticoes=n_repeticoes, metodo_norm=metodo)
        linhas.append({"Normalização": metodo, "F1_macro": res["f1_macro_media"], "F1_std": res["f1_macro_std"],
                       "Acurácia": res["acuracia_media"], "Acurácia_std": res["acuracia_std"]})
        print(f"  {metodo:8s} | F1 macro = {res['f1_macro_media']:.4f} | acurácia = {res['acuracia_media']:.4f}")
    pd.DataFrame(linhas).to_csv(os.path.join(PASTA_TABELAS, f"arvore_e1_normalizacao_cenario_{cenario}.csv"), index=False)


def experimento_poda(X_tr, y_tr, X_val, y_val, prof, cenario):
    """Poda com metade da validação e avalia na outra metade."""
    from sklearn.model_selection import train_test_split
    print("\n[Poda]")
    X_poda, X_aval, y_poda, y_aval = train_test_split(X_val, y_val, test_size=0.5,
                                                      random_state=SEMENTE, stratify=y_val)
    arv = ArvoreDecisao("gini", prof_max=20, min_folha=5).fit(X_tr, y_tr)
    linhas = []
    f_antes = f1_macro(y_aval, arv.predict(X_aval))
    linhas.append({"Árvore": "Sem poda", "N_nós": contar_nos(arv.raiz), "F1_macro": f_antes})
    arv_podada = copy.deepcopy(arv)
    podar(arv_podada.raiz, X_poda, y_poda)
    f_depois = f1_macro(y_aval, arv_podada.predict(X_aval))
    linhas.append({"Árvore": "Com poda", "N_nós": contar_nos(arv_podada.raiz), "F1_macro": f_depois})
    for l in linhas:
        print(f"  {l['Árvore']}: {l['N_nós']} nós | F1 macro = {l['F1_macro']:.4f}")
    pd.DataFrame(linhas).to_csv(os.path.join(PASTA_TABELAS, f"arvore_poda_cenario_{cenario}.csv"), index=False)


def experimento_e4(X_tr, y_tr, X_te, prof, cenario):
    """E4 (lado da árvore): tempo de treino e de predição de 1000 instâncias."""
    print("\n[E4 - Custo computacional da árvore]")
    amostra = X_te[:1000]
    if len(amostra) < 1000:   # teste pequeno: repete linhas até completar 1000
        idx = np.random.default_rng(SEMENTE).choice(len(X_te), size=1000, replace=True)
        amostra = X_te[idx]
    arv = ArvoreDecisao("gini", prof, 5)
    t0 = time.perf_counter()
    arv.fit(X_tr, y_tr)
    t_treino = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    arv.predict(amostra)
    t_pred = (time.perf_counter() - t0) * 1000
    print(f"  Treino: {t_treino:.1f} ms | Predição de 1000 instâncias: {t_pred:.1f} ms")
    pd.DataFrame([{"Modelo": "Árvore (própria)", "Tempo_Treino_ms": t_treino,
                   "Tempo_Predicao_1000_inst_ms": t_pred}]).to_csv(
        os.path.join(PASTA_TABELAS, f"arvore_e4_custo_cenario_{cenario}.csv"), index=False)


def experimento_e5(X_trval, y_trval, X_te, y_te, nomes_atributos, prof, f1_cv, cenario):
    """E5: avaliação final no teste (uma única vez) + regras + importância."""
    from meuknn import matriz_confusao, relatorio_metricas
    print("\n[E5 - Árvore: avaliação no teste]")
    arv = ArvoreDecisao("gini", prof, 5).fit(X_trval, y_trval)
    y_pred = arv.predict(X_te)
    M = matriz_confusao(np.asarray(y_te, dtype=int), y_pred, n_classes=3)
    metr = relatorio_metricas(M)
    print(f"  Acurácia: {metr['acuracia']*100:.2f}% | F1 macro: {metr['f1_macro']:.4f} "
          f"(na CV foi {f1_cv:.4f})")
    for i, nome in enumerate(ROTULOS):
        print(f"  {nome:8s} -> Precisão: {metr['precisao'][i]:.3f} | Revocação: {metr['revocacao'][i]:.3f} | F1: {metr['f1'][i]:.3f}")
    ie = ROTULOS.index("Enrolled")
    print(f"  Enrolled: {M[ie, ie]}/{M[ie].sum()} acertos | {M[ie, 0]} confundidos com Dropout | {M[ie, 2]} com Graduate")
    print("  Matriz de confusão (linhas = real, colunas = previsto):")
    print(M)

    pd.DataFrame(M, index=ROTULOS, columns=ROTULOS).to_csv(
        os.path.join(PASTA_TABELAS, f"arvore_matriz_confusao_cenario_{cenario}.csv"))
    pd.DataFrame({"Classe": ROTULOS, "Precisao": metr["precisao"], "Revocacao": metr["revocacao"],
                  "F1": metr["f1"]}).to_csv(
        os.path.join(PASTA_TABELAS, f"arvore_metricas_por_classe_cenario_{cenario}.csv"), index=False)

    regras = pd.DataFrame(extrair_regras(arv.raiz, list(nomes_atributos), ROTULOS))
    regras = regras.sort_values("cobertura", ascending=False)
    regras.to_csv(os.path.join(PASTA_TABELAS, f"arvore_regras_cenario_{cenario}.csv"), index=False)
    print("  Regras com maior cobertura:")
    for _, r in regras.head(3).iterrows():
        print(f"    {r['regra']} (cobertura {r['cobertura']*100:.1f}%, confiança {r['confianca']*100:.1f}%)")

    imp = importancia_atributos(arv.raiz, arv.n_atributos)
    tabela_imp = pd.DataFrame({"Atributo": list(nomes_atributos), "Importancia": imp})
    tabela_imp = tabela_imp.sort_values("Importancia", ascending=False)
    tabela_imp.to_csv(os.path.join(PASTA_TABELAS, f"arvore_importancia_cenario_{cenario}.csv"), index=False)
    print("  Atributos mais importantes:")
    for _, r in tabela_imp.head(5).iterrows():
        print(f"    {r['Atributo']:45s} {r['Importancia']*100:.1f}%")


def rodar_cenario(cenario, n_folds=10, n_repeticoes=3):
    from meuknn import preparar_referencia_e_alvo
    os.makedirs(PASTA_TABELAS, exist_ok=True)

    X_tr_df, X_val_df, X_te_df, y_tr, y_val, y_te = preparar_dados(cenario)
    print("\n" + "=" * 20)
    print(f"[Árvore - Cenário {cenario}]")
    print("=" * 20)
    print(f"Treino: {X_tr_df.shape} | Validação: {X_val_df.shape} | Teste: {X_te_df.shape}")

    X_tr, X_val = preparar_referencia_e_alvo(X_tr_df, X_val_df, "minmax")
    _, X_te_tr = preparar_referencia_e_alvo(X_tr_df, X_te_df, "minmax")

    comparar_com_sklearn(X_tr, y_tr, X_val, y_val, cenario)

    print("\n[Curva de profundidade]")
    df = curva_profundidade(X_tr, y_tr, X_val, y_val, profundidades=range(1, 16))
    df.to_csv(os.path.join(PASTA_TABELAS, f"arvore_curva_profundidade_cenario_{cenario}.csv"), index=False)
    plotar_curva(df, cenario)

    prof, f1_cv = escolher_profundidade_cv(X_tr_df, y_tr, n_folds, n_repeticoes)
    experimento_e1(X_tr_df, y_tr, prof, cenario, n_folds, n_repeticoes)
    experimento_poda(X_tr, y_tr, X_val, y_val, prof, cenario)
    experimento_e4(X_tr, y_tr, X_te_tr, prof, cenario)

    X_trval_df = pd.concat([X_tr_df, X_val_df])
    y_trval = np.concatenate([y_tr, y_val])
    X_trval, X_te = preparar_referencia_e_alvo(X_trval_df, X_te_df, "minmax")
    experimento_e5(X_trval, y_trval, X_te, y_te, X_trval_df.columns, prof, f1_cv, cenario)


# 7. Testes
def rodar_testes():
    # cálculo à mão
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    coluna = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert np.isclose(ganho_informacao(coluna, y, 0.5, "entropia"), 0.61, atol=1e-3)
    assert np.isclose(ganho_informacao(coluna, y, 0.5, "gini"), 1 / 3, atol=1e-3)
    print("Teste 1 (cálculo à mão): OK")

    # exemplo dos 8 alunos (colunas: bolsista, nota)
    X_ex = np.array([[0, 8], [0, 10], [0, 14], [0, 16],
                     [1, 9], [1, 15], [1, 17], [1, 18]], dtype=float)
    y_ex = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    arv = ArvoreDecisao("gini", prof_max=5, min_folha=1).fit(X_ex, y_ex)
    assert arv.raiz.atributo == 1 and np.isclose(arv.raiz.limiar, 12.0)
    assert contar_nos(arv.raiz) == 5
    assert list(arv.predict(np.array([[1, 9], [0, 11], [0, 15]], dtype=float))) == [1, 0, 1]
    print("Teste 2 (exemplo dos 8 alunos): OK")

    # sobreajuste e prof_max = 0
    rng = np.random.default_rng(0)
    Xa = rng.random((100, 3))
    ya = rng.integers(0, 3, size=100)
    arv = ArvoreDecisao("gini", prof_max=50, min_folha=1).fit(Xa, ya)
    assert np.mean(arv.predict(Xa) == ya) == 1.0
    assert contar_nos(ArvoreDecisao("gini", prof_max=0).fit(Xa, ya).raiz) == 1
    print("Teste 3 (sobreajuste): OK")

    # regras e importância
    arv = ArvoreDecisao("gini", prof_max=5, min_folha=1).fit(X_ex, y_ex)
    regras = extrair_regras(arv.raiz, ["Bolsista", "Nota"], ["Dropout", "Graduate"])
    for r in regras:
        print("   ", r["regra"], f"(cobertura {r['cobertura']*100:.0f}%, confiança {r['confianca']*100:.0f}%)")
    assert np.isclose(importancia_atributos(arv.raiz, 2).sum(), 1.0)
    print("Teste 4 (regras e importância): OK")

    # poda
    Xp = rng.normal(size=(600, 6))
    yp = ((Xp[:, 0] + 0.5 * Xp[:, 1] + rng.normal(size=600)) > 0).astype(int)
    X_t, y_t, X_v, y_v = Xp[:300], yp[:300], Xp[300:], yp[300:]
    arv = ArvoreDecisao("gini", prof_max=20, min_folha=1).fit(X_t, y_t)
    nos_antes = contar_nos(arv.raiz)
    acc_antes = np.mean(arv.predict(X_v) == y_v)
    podar(arv.raiz, X_v, y_v)
    acc_depois = np.mean(arv.predict(X_v) == y_v)
    assert contar_nos(arv.raiz) <= nos_antes and acc_depois >= acc_antes
    print(f"Teste 5 (poda): {nos_antes} -> {contar_nos(arv.raiz)} nós | acc {acc_antes:.3f} -> {acc_depois:.3f}: OK")

    # comparação com o sklearn
    try:
        from sklearn.tree import DecisionTreeClassifier
        minha = ArvoreDecisao("gini", prof_max=5, min_folha=5).fit(X_t, y_t)
        sk = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEMENTE).fit(X_t, y_t)
        a1, a2 = np.mean(minha.predict(X_v) == y_v), np.mean(sk.predict(X_v) == y_v)
        print(f"Teste 6: acurácia própria={a1:.3f} | sklearn={a2:.3f}")
        assert abs(a1 - a2) < 0.05
    except ImportError:
        print("Teste 6 pulado (sem scikit-learn)")

    print("\nTodos os testes passaram.")


if __name__ == "__main__":
    rodar_testes()
    for cen in sys.argv[1:]:
        rodar_cenario(cen.upper())