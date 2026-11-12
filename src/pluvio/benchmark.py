"""Roda o benchmark no período de DESENVOLVIMENTO e imprime as tabelas.

    python -m pluvio.benchmark                  # modo conservador (padrão)
    python -m pluvio.benchmark --modo hibrido

O período de teste (jul-out/2026) fica TRANCADO: este script corta a tabela em
FIM_DEV antes de calcular qualquer coisa. Abrir o teste é um passo separado,
feito uma vez, no fim, com tudo já decidido.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import avaliacao as av
from .dataset import climatologia_mensal, montar, observado_ate_emissao
from .inmet import ler_pasta
from .modelo import prever_andando
from .previsoes import ler_modelo
from .rotulo import LIMIAR_MM, rotular_dias

MODELOS = ["gfs_global", "icon_global", "ecmwf_ifs025"]
INICIO_DEV = "2024-02-05"      # primeiro dia com day2 arquivado nos três modelos
FIM_DEV = "2026-06-30"         # depois disso é teste trancado
FIM_CLIMA = "2023-12-31"       # climatologia só com dados anteriores ao período
PRIMEIRO_MES_PREVISTO = "2025-01"
RECALL_ALVO = 0.70


def tabela_dev(modo: str, pasta_inmet="data/raw/inmet", pasta_om="data/raw/openmeteo") -> pd.DataFrame:
    chuva = ler_pasta(pasta_inmet)
    rotulos = rotular_dias(chuva)
    previsoes = {m: ler_modelo(pasta_om, m) for m in MODELOS}
    t = montar(rotulos, previsoes, modo)
    t["obs_vespera"] = observado_ate_emissao(chuva)
    t["clima_mes"] = t.index.month.map(climatologia_mensal(rotulos, FIM_CLIMA))
    return t.loc[INICIO_DEV:FIM_DEV]      # <- o cadeado do teste


def com_features(t: pd.DataFrame) -> pd.DataFrame:
    prev = t[[f"prev_{m}" for m in MODELOS]]
    t = t.copy()
    # log(1+mm): a logística é linear na entrada e chuva tem cauda longa;
    # ir de 0 para 10 mm importa mais que de 40 para 50.
    t["f_max"] = np.log1p(prev.max(axis=1, skipna=False))
    t["f_min"] = np.log1p(prev.min(axis=1, skipna=False))
    t["f_cape"] = np.log1p(t["cape_gfs_global"])
    t["f_clima"] = t["clima_mes"]
    for m in MODELOS:
        t[f"f_{m}"] = np.log1p(t[f"prev_{m}"])
    return t


def linha(nome, evento, pontuacao, prob=None, clima=None) -> dict:
    op = av.ponto_de_operacao(evento, pontuacao, RECALL_ALVO)
    n_ev = int(np.sum(evento))
    baixo, alto = av.wilson(op.get("acertos", 0), n_ev)
    return {
        "previsor": nome,
        "prec_media": round(av.precisao_media(evento, pontuacao), 3),
        "limiar": round(op.get("limiar", np.nan), 3),
        "pegou": f'{op.get("acertos", 0)}/{n_ev}',
        "recall_ic95": f"{baixo:.0%}-{alto:.0%}",
        "alarmes_falsos": op.get("alarmes_falsos"),
        "precisao": round(op.get("precisao", np.nan), 2),
        "brier_skill": round(av.brier_skill(evento, prob, clima), 3) if prob is not None else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", default="conservador", choices=["conservador", "hibrido"])
    args = ap.parse_args()

    t = com_features(tabela_dev(args.modo))
    usadas = ["evento", "obs_vespera", "f_max", "f_min", "f_cape", "f_clima"] + [f"f_{m}" for m in MODELOS]
    limpa = t.dropna(subset=usadas).copy()
    limpa["evento"] = limpa["evento"].astype(bool)
    print(f"modo={args.modo} | dias no período: {len(t)} | completos: {len(limpa)} "
          f"| eventos: {int(limpa.evento.sum())} ({limpa.evento.mean():.1%})")
    print("eventos por ano:", limpa.groupby(limpa.index.year).evento.agg(["sum", "size"]).to_dict("index"))
    print("\ncorrelação entre features (treino até 2024-12):")
    print(limpa.loc[:"2024-12-31", ["f_max", "f_min", "f_cape", "f_clima"]].corr().round(2).to_string())

    # ---- A. tudo no período inteiro, sem treino (limiar escolhido olhando o próprio período)
    ev = limpa["evento"].to_numpy()
    print(f"\nA. Período inteiro. Ponto de operação: limiar mais alto com recall >= {RECALL_ALVO:.0%}")
    linhas = [linha(f"{m} (mm em 3h)", ev, limpa[f"prev_{m}"]) for m in MODELOS]
    linhas.append(linha("máximo dos 3", ev, limpa[[f"prev_{m}" for m in MODELOS]].max(axis=1)))
    linhas.append(linha("persistência (obs. até 21h)", ev, limpa["obs_vespera"]))
    linhas.append(linha("climatologia do mês", ev, limpa["clima_mes"]))
    print(pd.DataFrame(linhas).drop(columns="brier_skill").to_string(index=False))
    for m in MODELOS:
        c = av.contingencia(ev, limpa[f"prev_{m}"] >= LIMIAR_MM)
        print(f"   {m} com o limiar ingênuo de {LIMIAR_MM:.0f} mm: pegou {c['acertos']}/{int(ev.sum())}, "
              f"{c['alarmes_falsos']} alarmes falsos")

    # ---- B. janela expansível: tudo aprendido só com o passado
    print(f"\nB. Janela expansível, previsões de {PRIMEIRO_MES_PREVISTO} em diante")
    variantes = {
        "logística: só clima": ["f_clima"],
        **{f"logística: só {m}": [f"f_{m}"] for m in MODELOS},
        "logística: max+min+cape+clima": ["f_max", "f_min", "f_cape", "f_clima"],
    }
    probs = {nome: prever_andando(limpa, cols, PRIMEIRO_MES_PREVISTO) for nome, cols in variantes.items()}
    fora = limpa.loc[probs["logística: só clima"].index]
    ev2, clima2 = fora["evento"].to_numpy(), fora["clima_mes"].to_numpy()
    linhas = [linha("climatologia 2006-2023", ev2, clima2, clima2, clima2)]
    linhas += [linha(nome, ev2, p.to_numpy(), p.to_numpy(), clima2) for nome, p in probs.items()]
    print(f"   dias: {len(fora)} | eventos: {int(ev2.sum())}")
    print(pd.DataFrame(linhas).to_string(index=False))


if __name__ == "__main__":
    main()
