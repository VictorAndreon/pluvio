"""Métricas do benchmark, escritas à mão para dar para ler o que cada uma faz.

Vocabulário (tabela de contingência de um alerta sim/não):

                      choveu forte        não choveu forte
    alertou           acerto (a)          alarme falso (b)
    não alertou       perda (c)           acerto negativo (d)

    recall (POD)  = a / (a + c)   "dos temporais reais, quantos eu peguei?"
    precisão      = a / (a + b)   "dos meus alertas, quantos eram reais?"
    CSI           = a / (a+b+c)   acerto ignorando os dias calmos (d), que são
                                  ~95% e inflam qualquer "acurácia"

Com ~5% de dias positivos, ACURÁCIA NÃO SERVE: dizer "nunca chove forte"
acerta 95% dos dias e pega zero temporais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def contingencia(evento, alerta) -> dict:
    evento = np.asarray(evento, dtype=bool)
    alerta = np.asarray(alerta, dtype=bool)
    a = int((alerta & evento).sum())
    b = int((alerta & ~evento).sum())
    c = int((~alerta & evento).sum())
    d = int((~alerta & ~evento).sum())
    return {
        "acertos": a,
        "alarmes_falsos": b,
        "perdas": c,
        "negativos": d,
        "recall": a / (a + c) if a + c else np.nan,
        "precisao": a / (a + b) if a + b else np.nan,
        "csi": a / (a + b + c) if a + b + c else np.nan,
    }


def wilson(sucessos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança (95% por padrão) para uma proporção.

    Serve para lembrar o tamanho da incerteza: 8 acertos em 12 dá recall 67%,
    mas o intervalo vai de ~39% a ~86%.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = sucessos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    meia = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centro - meia, centro + meia)


def brier(evento, probabilidade) -> float:
    """Erro quadrático médio da probabilidade prevista. 0 = perfeito.

    Usa TODOS os dias (não só os eventos), por isso é mais estável que o
    recall quando há poucos eventos.
    """
    evento = np.asarray(evento, dtype=float)
    probabilidade = np.asarray(probabilidade, dtype=float)
    return float(np.mean((probabilidade - evento) ** 2))


def brier_skill(evento, probabilidade, referencia) -> float:
    """Quanto do erro da referência foi eliminado. >0 = melhor que a referência.

    1 = perfeito, 0 = igual à referência, negativo = pior que ela.
    """
    return 1.0 - brier(evento, probabilidade) / brier(evento, referencia)


def curva_de_limiares(evento, pontuacao) -> pd.DataFrame:
    """Para cada limiar possível, o que aconteceria se alertássemos quando
    `pontuacao >= limiar`. É a curva precisão-recall em forma de tabela.

    A pontuação pode ser mm previstos por um modelo ou uma probabilidade.
    """
    evento = np.asarray(evento, dtype=bool)
    pontuacao = np.asarray(pontuacao, dtype=float)
    linhas = []
    for limiar in np.unique(pontuacao)[::-1]:
        linha = contingencia(evento, pontuacao >= limiar)
        linha["limiar"] = float(limiar)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def precisao_media(evento, pontuacao) -> float:
    """Área sob a curva precisão-recall (average precision).

    Mede só a ORDENAÇÃO: os dias de evento ficaram no topo da lista? Não
    depende de escolher limiar. Referência: uma pontuação aleatória dá a taxa
    base (~0,05 aqui); 1,0 é perfeito.
    """
    evento = np.asarray(evento, dtype=bool)
    ordem = np.argsort(-np.asarray(pontuacao, dtype=float), kind="stable")
    acertou = evento[ordem]
    precisao_no_k = np.cumsum(acertou) / (np.arange(len(acertou)) + 1)
    return float(precisao_no_k[acertou].mean()) if acertou.any() else np.nan


def ponto_de_operacao(evento, pontuacao, recall_minimo: float) -> dict:
    """O limiar MAIS ALTO que ainda pega pelo menos `recall_minimo` dos eventos.

    Traduz a prioridade do projeto ("perder temporal é pior que alarme falso,
    mas com limite") num número: fixado o recall, quantos alarmes falsos custa?
    """
    curva = curva_de_limiares(evento, pontuacao)
    ok = curva[curva["recall"] >= recall_minimo]
    return ok.iloc[0].to_dict() if len(ok) else {}


def bootstrap_em_blocos(datas, estatistica, n_reamostras: int = 2000, dias_por_bloco: int = 7,
                        semente: int = 0) -> tuple[float, float]:
    """Intervalo de 95% para `estatistica`, reamostrando SEMANAS inteiras.

    Bootstrap = "e se o período tivesse sido outro?": sorteia blocos com
    reposição, recalcula a estatística, repete milhares de vezes e olha o
    espalhamento do resultado.

    Por que blocos e não dias soltos: dias vizinhos são parecidos (a mesma
    frente dura dias, e várias estações veem o mesmo sistema). Sortear dias
    soltos trataria 3 dias da mesma frente como 3 evidências independentes e
    daria um intervalo estreito demais. Sorteando o bloco inteiro, o que anda
    junto continua junto.

    `datas`       : data de cada linha (pode repetir, uma por estação).
    `estatistica` : função que recebe um vetor de posições (inteiros) e
                    devolve um número.
    """
    datas = pd.DatetimeIndex(datas)
    bloco = (datas - datas.min()).days // dias_por_bloco
    posicoes_por_bloco = [np.flatnonzero(bloco == b) for b in np.unique(bloco)]
    rng = np.random.default_rng(semente)
    valores = []
    for _ in range(n_reamostras):
        sorteio = rng.integers(0, len(posicoes_por_bloco), len(posicoes_por_bloco))
        valores.append(estatistica(np.concatenate([posicoes_por_bloco[i] for i in sorteio])))
    baixo, alto = np.nanpercentile(valores, [2.5, 97.5])
    return float(baixo), float(alto)
