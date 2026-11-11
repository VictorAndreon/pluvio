"""Rótulo do projeto: "o dia D teve alguma janela de 3h seguidas com >= 20 mm?"

Convenções (cada uma é uma decisão; mudar qualquer uma muda a contagem):

1. HORA = FIM DO PERÍODO. O registro das 15:00 UTC é a chuva acumulada entre
   14:00 e 15:00 UTC. (Premissa sobre o INMET; a Open-Meteo documenta a mesma
   convenção para `precipitation`, então as duas fontes alinham sem deslocar.)

2. DIA LOCAL = UTC-3 FIXO, sem horário de verão. Passo Fundo teve horário de
   verão até 2019; usar o relógio "de parede" faria o dia ter 23h ou 25h duas
   vezes por ano. UTC-3 fixo dá sempre 24h e é o fuso real de 2019 em diante.
   Juntando com (1): o dia local D são os registros de 04:00 UTC de D até
   03:00 UTC de D+1.

3. A JANELA PERTENCE AO DIA EM QUE TERMINA. A janela 22h-01h conta para o dia
   seguinte. Assim cada janela tem exatamente um dono e nenhum temporal de
   virada de dia some. Consequência: o rótulo de D enxerga chuva a partir das
   22h de D-1 -- ainda depois da emissão das 21h, então não há vazamento.

4. HORA FALTANDO = "NÃO SEI". Janela com qualquer hora ausente não tem soma.
   O dia vale True se ALGUMA janela completa atingiu o limiar (isso é certo
   mesmo com buracos), False só se TODAS as 24 janelas estão completas, e
   NA (desconhecido) caso contrário. Chutar False num dia com buraco
   ensinaria ao modelo que "sensor quebrado" significa "não choveu".
"""
from __future__ import annotations

import pandas as pd

LIMIAR_MM = 20.0
JANELA_H = 3
FUSO_LOCAL = pd.Timedelta(hours=-3)


def soma_movel(chuva_utc: pd.Series, janela_h: int = JANELA_H) -> pd.Series:
    """Soma das `janela_h` horas que terminam em cada instante (NaN se faltar alguma).

    Exige índice horário sem buracos (ver `inmet.ler_pasta`), porque a janela
    é contada em linhas.
    """
    if chuva_utc.index.freq is None or chuva_utc.index.freq != pd.tseries.frequencies.to_offset("h"):
        raise ValueError("a série precisa ter frequência horária contínua (use asfreq('h'))")
    return chuva_utc.rolling(janela_h, min_periods=janela_h).sum()


def dia_local(instantes_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dia local (UTC-3 fixo) ao qual pertence o período que TERMINA em cada instante.

    Tira 1 hora (para olhar o início do período, convenção 1) e mais 3 (fuso).
    Ex.: 03:00 UTC do dia 2 = período 23h-24h local do dia 1 -> dia 1.
    """
    inicio_local = instantes_utc - pd.Timedelta(hours=1) + FUSO_LOCAL
    return inicio_local.tz_localize(None).normalize()


def rotular_dias(
    chuva_utc: pd.Series, limiar_mm: float = LIMIAR_MM, janela_h: int = JANELA_H
) -> pd.DataFrame:
    """Uma linha por dia local, com:

    evento        : True / False / <NA>   (ver convenção 4)
    max_3h_mm     : maior soma entre as janelas completas do dia (NaN se nenhuma)
    janelas_ok    : quantas das 24 janelas do dia estão completas
    """
    soma = soma_movel(chuva_utc, janela_h)
    dia = dia_local(soma.index)
    grupos = soma.groupby(dia)

    saida = pd.DataFrame(
        {
            "max_3h_mm": grupos.max(),
            "janelas_ok": grupos.count(),   # count ignora NaN
            "janelas_total": grupos.size(),
        }
    )
    teve = saida["max_3h_mm"] >= limiar_mm
    completo = saida["janelas_ok"] == 24

    evento = pd.Series(pd.NA, index=saida.index, dtype="boolean")
    evento[teve] = True
    evento[~teve & completo] = False
    saida["evento"] = evento
    saida.index.name = "dia"
    # dias nas pontas da série têm menos de 24 registros; ficam como estão
    # (NA, a menos que um evento tenha sido observado).
    return saida[["evento", "max_3h_mm", "janelas_ok"]]
