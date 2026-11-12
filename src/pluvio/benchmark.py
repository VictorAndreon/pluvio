"""Roda o benchmark no período de DESENVOLVIMENTO e imprime as tabelas.

    python -m pluvio.benchmark                  # modo conservador (padrão)
    python -m pluvio.benchmark --modo hibrido

O período de teste (jul-out/2026) fica TRANCADO: este script corta a tabela em
FIM_DEV antes de calcular qualquer coisa. Abrir o teste é um passo separado,
feito uma vez, no fim, com tudo já decidido.

Usa todas as estações que tenham CSV do INMET e previsões baixadas. Cada linha
da tabela é um par (estação, dia).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import avaliacao as av
from .dataset import climatologia_mensal, montar, observado_ate_emissao
from .inmet import ler_pasta, listar_estacoes
from .modelo import prever_andando
from .previsoes import ler_modelo
from .rotulo import LIMIAR_MM, rotular_dias

MODELOS = ["gfs_global", "icon_global", "ecmwf_ifs025"]
ESTACAO_FOCO = "A839"          # Passo Fundo
INICIO_DEV = "2024-02-05"      # primeiro dia com day2 arquivado nos três modelos
FIM_DEV = "2026-06-30"         # depois disso é teste trancado
FIM_CLIMA = "2023-12-31"       # climatologia só com dados anteriores ao período
PRIMEIRO_MES_PREVISTO = "2025-01"
RECALL_ALVO = 0.70

# Entradas do modelo combinado, fixadas por argumento ANTES de medir desempenho:
#   f_max   : maior chuva em 3h prevista entre os três modelos (não perder o
#             aviso de um modelo isolado).
#   f_cape  : CAPE do GFS, "combustível" para tempestade; informação que a
#             chuva prevista não carrega.
#   f_clima : frequência histórica de eventos no mês.
# A "discordância entre modelos" do plano original (e o mínimo, que testei no
# lugar) saiu: correlação de 0,86 com f_max no treino. Duas entradas quase
# iguais com ~40 eventos deixam os coeficientes instáveis.
COMBINADO = ["f_max", "f_cape", "f_clima"]


def tabela_dev(modo: str, pasta_inmet="data/raw/inmet", pasta_om="data/raw/openmeteo") -> pd.DataFrame:
    # Climatologia regional: a de Passo Fundo (20 anos de série) vale para todas.
    clima = climatologia_mensal(rotular_dias(ler_pasta(pasta_inmet, ESTACAO_FOCO)), FIM_CLIMA)
    partes = []
    for codigo in listar_estacoes(pasta_inmet):
        if not (Path(pasta_om) / codigo).is_dir():
            continue                       # estação sem previsões baixadas
        chuva = ler_pasta(pasta_inmet, codigo)
        previsoes = {m: ler_modelo(pasta_om, m, codigo) for m in MODELOS}
        t = montar(rotular_dias(chuva), previsoes, modo)
        t["obs_vespera"] = observado_ate_emissao(chuva)
        t["clima_mes"] = t.index.month.map(clima)
        t["estacao"] = codigo
        partes.append(t.loc[INICIO_DEV:FIM_DEV])      # <- o cadeado do teste
    return pd.concat(partes).sort_index()


def com_features(t: pd.DataFrame) -> pd.DataFrame:
    prev = t[[f"prev_{m}" for m in MODELOS]]
    t = t.copy()
    # log(1+mm): a logística é linear na entrada e chuva tem cauda longa;
    # ir de 0 para 10 mm importa mais que de 40 para 50.
    t["f_max"] = np.log1p(prev.max(axis=1, skipna=False))
    t["f_cape"] = np.log1p(t["cape_gfs_global"])
    t["f_clima"] = t["clima_mes"]
    for m in MODELOS:
        t[f"f_{m}"] = np.log1p(t[f"prev_{m}"])
    return t


def linha_a(nome, evento, pontuacao) -> dict:
    op = av.ponto_de_operacao(evento, pontuacao, RECALL_ALVO)
    return {
        "previsor": nome,
        "dias": len(evento),
        "prec_media": round(av.precisao_media(evento, pontuacao), 3),
        "limiar": round(op.get("limiar", np.nan), 2),
        "pegou": f'{int(op.get("acertos", 0))}/{int(np.sum(evento))}',
        "alarmes_falsos": int(op.get("alarmes_falsos", 0)),
        "precisao": round(op.get("precisao", np.nan), 2),
    }


def linha_b(nome, fora: pd.DataFrame, saida: pd.DataFrame | None) -> dict:
    ev, clima = fora["evento"].to_numpy(), fora["clima_mes"].to_numpy()
    if saida is None:        # a própria climatologia: não emite alerta sim/não
        return {"previsor": nome, "prec_media": round(av.precisao_media(ev, clima), 3), "brier_skill": 0.0}
    p = saida["prob"].to_numpy()
    c = av.contingencia(ev, saida["alerta"].to_numpy())
    baixo, alto = av.wilson(c["acertos"], int(ev.sum()))
    bs_baixo, bs_alto = av.bootstrap_em_blocos(
        fora.index, lambda i: av.brier_skill(ev[i], p[i], clima[i]))
    return {
        "previsor": nome,
        "prec_media": round(av.precisao_media(ev, p), 3),
        "brier_skill": round(av.brier_skill(ev, p, clima), 3),
        "bs_ic95": f"{bs_baixo:+.2f} a {bs_alto:+.2f}",
        "pegou": f'{c["acertos"]}/{int(ev.sum())}',
        "recall_ic95": f"{baixo:.0%}-{alto:.0%}",
        "alarmes_falsos": c["alarmes_falsos"],
        "precisao": round(c["precisao"], 2),
    }


def secao_a(limpa: pd.DataFrame, t: pd.DataFrame) -> None:
    ev = limpa["evento"].to_numpy()
    linhas = [linha_a(f"{m} (mm em 3h)", ev, limpa[f"prev_{m}"]) for m in MODELOS]
    linhas.append(linha_a("máximo dos 3", ev, limpa[[f"prev_{m}" for m in MODELOS]].max(axis=1)))
    linhas.append(linha_a("climatologia do mês", ev, limpa["clima_mes"]))
    # a persistência tem a própria lista de dias: precisa da véspera observada,
    # coisa que os outros previsores não exigem
    pers = t.dropna(subset=["evento", "obs_vespera"])
    linhas.append(linha_a("persistência (obs. até 21h)", pers["evento"].astype(bool).to_numpy(), pers["obs_vespera"]))
    print(pd.DataFrame(linhas).to_string(index=False))
    for m in MODELOS:
        c = av.contingencia(ev, limpa[f"prev_{m}"] >= LIMIAR_MM)
        print(f"   {m} com o limiar ingênuo de {LIMIAR_MM:.0f} mm: pegou {c['acertos']}/{int(ev.sum())}, "
              f"{c['alarmes_falsos']} alarmes falsos")


def secao_b(limpa: pd.DataFrame) -> None:
    variantes = {**{f"logística: só {m}": [f"f_{m}"] for m in MODELOS},
                 "logística: combinado (max+cape+clima)": COMBINADO}
    saidas = {nome: prever_andando(limpa.set_index("dia"), cols, PRIMEIRO_MES_PREVISTO, recall_alvo=RECALL_ALVO)
              for nome, cols in variantes.items()}
    fora = limpa[limpa["dia"] >= pd.Period(PRIMEIRO_MES_PREVISTO, freq="M").start_time].set_index("dia")
    for rotulo, mascara in [("todas as estações", np.ones(len(fora), bool)),
                            (f"só {ESTACAO_FOCO}", (fora["estacao"] == ESTACAO_FOCO).to_numpy())]:
        if mascara.all() and rotulo.startswith("só"):
            continue
        sub = fora[mascara]
        print(f"\n   [{rotulo}] dias: {len(sub)} | eventos: {int(sub.evento.sum())}")
        linhas = [linha_b("climatologia 2006-2023", sub, None)]
        linhas += [linha_b(nome, sub, s[mascara]) for nome, s in saidas.items()]
        print(pd.DataFrame(linhas).fillna("").to_string(index=False))

        # a pergunta do projeto: combinar ganha de cada modelo sozinho?
        ev, clima = sub["evento"].to_numpy(), sub["clima_mes"].to_numpy()
        pc = saidas["logística: combinado (max+cape+clima)"]["prob"].to_numpy()[mascara]
        for m in MODELOS:
            pm = saidas[f"logística: só {m}"]["prob"].to_numpy()[mascara]
            dif = lambda i: av.brier_skill(ev[i], pc[i], clima[i]) - av.brier_skill(ev[i], pm[i], clima[i])
            baixo, alto = av.bootstrap_em_blocos(sub.index, dif)
            veredito = "ganho" if baixo > 0 else "perda" if alto < 0 else "inconclusivo"
            print(f"   combinado − só {m}: {dif(np.arange(len(ev))):+.3f} de Brier skill "
                  f"(IC95 {baixo:+.3f} a {alto:+.3f}) -> {veredito}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", default="conservador", choices=["conservador", "hibrido"])
    args = ap.parse_args()

    t = com_features(tabela_dev(args.modo))
    usadas = ["evento"] + COMBINADO + [f"f_{m}" for m in MODELOS]
    limpa = t.dropna(subset=usadas).copy()
    limpa["evento"] = limpa["evento"].astype(bool)
    limpa["dia"] = limpa.index
    print(f"modo={args.modo} | estações: {sorted(t.estacao.unique())} | linhas: {len(t)} "
          f"| completas: {len(limpa)} | eventos: {int(limpa.evento.sum())} ({limpa.evento.mean():.1%})")
    print("eventos/linhas por ano:", limpa.groupby(limpa.index.year).evento.agg(["sum", "size"]).to_dict("index"))

    print(f"\nA. Período inteiro, limiar escolhido no próprio período (o mais alto com recall >= {RECALL_ALVO:.0%})")
    secao_a(limpa, t)
    print(f"\nB. Janela expansível de {PRIMEIRO_MES_PREVISTO} em diante: modelo E limiar aprendidos só no passado")
    secao_b(limpa)


if __name__ == "__main__":
    main()
