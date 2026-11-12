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


def limiar_para_recall(evento, prob, recall_minimo: float) -> float:
    """Maior limiar de probabilidade que ainda pega `recall_minimo` dos eventos dados."""
    dos_eventos = np.sort(np.asarray(prob)[np.asarray(evento, dtype=bool)])[::-1]
    quantos = int(np.ceil(recall_minimo * len(dos_eventos)))
    return float(dos_eventos[quantos - 1])


def prever_andando(tabela: pd.DataFrame, colunas: list[str], primeiro_mes_previsto: str,
                   alvo: str = "evento", recall_alvo: float = 0.70) -> pd.DataFrame:
    """Previsão fora-da-amostra para cada dia a partir de `primeiro_mes_previsto`.

    Devolve duas colunas:
      prob   : probabilidade dada por um modelo treinado só com meses anteriores
      alerta : prob >= limiar, com o limiar TAMBÉM escolhido só no treino
               (o mais alto que pegaria `recall_alvo` dos eventos de treino).

    Escolher o limiar depois, olhando as previsões do período avaliado, daria
    uma contagem de alarmes falsos otimista: seria usar o gabarito para
    decidir onde cortar.

    `tabela` não pode ter NaN em `colunas` nem em `alvo` (filtre antes, de
    propósito: decidir o que fazer com dia incompleto é decisão sua, não do código).
    O índice pode ter datas repetidas (várias estações no mesmo dia): a divisão
    é por mês do calendário, então o mesmo dia nunca cai em lados diferentes.
    """
    if tabela[colunas + [alvo]].isna().any().any():
        raise ValueError("há NaN nas colunas usadas; filtre a tabela antes")
    datas = tabela.index
    meses = datas.to_period("M")
    partes = []
    for mes in sorted(meses.unique()):
        if mes < pd.Period(primeiro_mes_previsto, freq="M"):
            continue
        treino, previsto = tabela[meses < mes], tabela[meses == mes]
        y = treino[alvo].astype(int)
        if y.nunique() < 2:
            continue   # sem evento no treino não há o que aprender
        m = novo_modelo().fit(treino[colunas], y)
        limiar = limiar_para_recall(y, m.predict_proba(treino[colunas])[:, 1], recall_alvo)
        p = m.predict_proba(previsto[colunas])[:, 1]
        partes.append(pd.DataFrame({"prob": p, "alerta": p >= limiar}, index=previsto.index))
    return pd.concat(partes)
