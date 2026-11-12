"""Monta a tabela diária: uma linha por dia-alvo, só com o que existia às 21h da véspera.

A PERGUNTA: às 21h (local) do dia D-1, "o dia D vai ter evento?".
Em UTC: a emissão é às 00:00 UTC de D.

O QUE O RÓTULO DE D ENXERGA (ver rotulo.py): janelas de 3h que terminam em D,
isto é, registros horários de 02:00 UTC de D até 03:00 UTC de D+1.

O PROBLEMA COM `_previous_day1`
A Open-Meteo define `_previous_day1` como "o valor previsto 24h antes da
hora-alvo". É uma defasagem POR HORA, não "a rodada que existia às 21h":

    hora-alvo T              previsto em (T - 24h)       emissão (00 UTC de D)
    12:00 UTC de D           12:00 UTC de D-1            antes  -> ok
    00:00 UTC de D+1         00:00 UTC de D              empate -> ok só no papel
    03:00 UTC de D+1         03:00 UTC de D              DEPOIS -> vazamento

E há uma dúvida que a documentação não resolve: "previsto em T-24h" quer dizer
rodada INICIADA até T-24h ou PUBLICADA até T-24h? Uma rodada leva horas para
sair. Se for "iniciada", até o caso "ok" acima pode usar uma rodada que ainda
não estava publicada às 21h.

Por isso existem dois modos, e o projeto reporta os dois:

  "conservador" : usa `_previous_day2` (previsto 48h antes) para todas as horas.
                  Tudo foi previsto pelo menos 21h antes da emissão: limpo sob
                  qualquer interpretação. É um PISO: subestima o que dá para
                  fazer de verdade, porque joga fora a rodada mais recente.
  "hibrido"     : `_previous_day1` enquanto T - 24h <= emissão, `_previous_day2`
                  depois. Mais perto do uso real, mas depende da interpretação
                  acima. É um TETO otimista.

A verdade operacional fica entre os dois, e só a previsão ao vivo mede.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rotulo import JANELA_H, dia_local, soma_movel

MODOS = ("conservador", "hibrido")
HORA_EMISSAO_UTC = 0   # 21h local (UTC-3) = 00:00 UTC do dia-alvo


def serie_disponivel(tabela: pd.DataFrame, variavel: str, modo: str) -> pd.Series:
    """Para cada hora-alvo, o valor de `variavel` que respeita o modo escolhido."""
    if modo not in MODOS:
        raise ValueError(f"modo deve ser um de {MODOS}")
    dia2 = tabela[f"{variavel}_previous_day2"]
    if modo == "conservador":
        return dia2.rename(variavel)

    dia1 = tabela[f"{variavel}_previous_day1"]
    alvo = tabela.index
    # emissão que responde por esta hora: 00:00 UTC do dia local a que ela pertence
    emissao = dia_local(alvo).tz_localize("UTC") + pd.Timedelta(hours=HORA_EMISSAO_UTC)
    previsto_em = alvo - pd.Timedelta(hours=24)
    pode_dia1 = previsto_em <= emissao
    return dia1.where(pode_dia1, dia2).rename(variavel)


def resumo_diario(serie_horaria: pd.Series, como: str) -> pd.Series:
    """Resume uma série horária por dia local. Dia com hora faltando vira NaN.

    como="max_3h" : maior soma de 3h entre as 24 janelas que terminam no dia
                    (as MESMAS janelas do rótulo -- comparar maçã com maçã).
    como="max"    : maior valor horário entre as 24 horas do dia.
    """
    if como == "max_3h":
        base = soma_movel(serie_horaria, JANELA_H)
    elif como == "max":
        base = serie_horaria
    else:
        raise ValueError(como)
    grupos = base.groupby(dia_local(base.index))
    valor = grupos.max()
    completo = grupos.count() == 24
    return valor.where(completo)


def montar(rotulos: pd.DataFrame, modelos: dict[str, pd.DataFrame], modo: str) -> pd.DataFrame:
    """Tabela diária com o rótulo e, por modelo, a chuva máxima prevista em 3h.

    `modelos` é {nome: tabela horária} vinda de previsoes.ler_modelo.
    Colunas: evento, max_3h_mm (observado), prev_<modelo>, e cape_<modelo>
    quando o modelo tem CAPE arquivado.
    """
    saida = rotulos[["evento", "max_3h_mm"]].copy()
    for nome, tabela in modelos.items():
        chuva = serie_disponivel(tabela, "precipitation", modo)
        saida[f"prev_{nome}"] = resumo_diario(chuva, "max_3h")
        if tabela.get("cape_previous_day2") is not None and tabela["cape_previous_day2"].notna().any():
            cape = serie_disponivel(tabela, "cape", modo)
            saida[f"cape_{nome}"] = resumo_diario(cape, "max")
    return saida


def observado_ate_emissao(chuva_utc: pd.Series) -> pd.Series:
    """Maior soma de 3h observada na VÉSPERA até as 21h, indexada pelo dia-alvo.

    É a versão honesta de "choveu forte hoje?". Às 21h o dia ainda não
    acabou: só existem as janelas que terminam até 21h local (00:00 UTC do
    dia-alvo). São 21 das 24 janelas da véspera; se faltar alguma, NaN.
    Usar o rótulo do dia inteiro aqui seria vazamento (3 horas de futuro).
    """
    soma = soma_movel(chuva_utc, JANELA_H)
    dia = dia_local(soma.index)
    hora_local_fim = (soma.index + pd.Timedelta(hours=-3)).hour   # 0 = meia-noite = fim do dia
    ate_21h = (hora_local_fim >= 1) & (hora_local_fim <= 21)
    visivel = soma[ate_21h]
    grupos = visivel.groupby(dia[ate_21h])
    valor = grupos.max().where(grupos.count() == 21)
    valor.index = valor.index + pd.Timedelta(days=1)   # véspera -> dia-alvo
    return valor.rename("obs_vespera_ate_21h_mm")


def climatologia_mensal(rotulos: pd.DataFrame, ate: str) -> pd.Series:
    """Fração de dias com evento em cada mês do ano, usando só dias até `ate`.

    `ate` precisa ser anterior ao período de treino/teste (usamos 2023-12-31):
    assim a climatologia nunca contém um dia que o modelo vai tentar prever.
    """
    passado = rotulos.loc[:ate, "evento"].dropna().astype(float)
    return passado.groupby(passado.index.month).mean().rename("clima_mes")
