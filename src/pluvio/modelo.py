"""Regressão logística avaliada em janela expansível (walk-forward).

POR QUE JANELA EXPANSÍVEL
Sortear dias para treino e teste (o `train_test_split` padrão) vaza: o dia 14
no treino e o dia 15 no teste fazem parte da mesma frente fria. O modelo
"prevê" o dia 15 lembrando do 14. Aqui cada mês é previsto por um modelo que
só viu os meses ANTERIORES -- exatamente como seria usado na vida real:

    treina jan/24..dez/24  -> prevê jan/25
    treina jan/24..jan/25  -> prevê fev/25
    ...

Cada dia recebe uma probabilidade vinda de um modelo que nunca o viu, e as
métricas são calculadas juntando todas essas previsões.

POR QUE REGRESSÃO LOGÍSTICA SEM AJUSTE DE HIPERPARÂMETRO
Com ~40 eventos, cada escolha feita olhando resultado (C, features, limiar)
consome evidência. Tudo aqui é fixado antes: C=1.0, padronização aprendida
só no treino, nenhuma busca.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def novo_modelo():
    # O StandardScaler fica DENTRO do pipeline: média e desvio são calculados
    # só com as linhas de treino de cada rodada. Padronizar a tabela inteira
    # antes de dividir seria vazamento (a média conteria o futuro).
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def dobras(indice: pd.DatetimeIndex, primeiro_mes_previsto: str):
    """Gera (dias_de_treino, dias_previstos) mês a mês, treino sempre no passado."""
    meses = indice.to_period("M")
    for mes in sorted(meses.unique()):
        if mes < pd.Period(primeiro_mes_previsto, freq="M"):
            continue
        yield indice[meses < mes], indice[meses == mes]


def prever_andando(tabela: pd.DataFrame, colunas: list[str], primeiro_mes_previsto: str,
                   alvo: str = "evento") -> pd.Series:
    """Probabilidade fora-da-amostra para cada dia a partir de `primeiro_mes_previsto`.

    `tabela` não pode ter NaN em `colunas` nem em `alvo` (filtre antes, de
    propósito: decidir o que fazer com dia incompleto é decisão sua, não do código).
    """
    if tabela[colunas + [alvo]].isna().any().any():
        raise ValueError("há NaN nas colunas usadas; filtre a tabela antes")
    partes = []
    for treino, previsto in dobras(tabela.index, primeiro_mes_previsto):
        y = tabela.loc[treino, alvo].astype(int)
        if y.nunique() < 2:
            continue   # sem evento no treino não há o que aprender
        m = novo_modelo().fit(tabela.loc[treino, colunas], y)
        p = m.predict_proba(tabela.loc[previsto, colunas])[:, 1]
        partes.append(pd.Series(p, index=previsto))
    return pd.concat(partes).rename("prob")
