"""Leitura dos CSVs anuais do INMET (estação automática A839, Passo Fundo).

O INMET mudou o formato em 2019, então este módulo existe para devolver UMA
série horária limpa, independente do ano:

    índice : DatetimeIndex em UTC, de hora em hora, SEM buracos
    valor  : chuva da hora em mm; NaN = "não sei" (nunca zero)

Diferenças entre os formatos (verificadas nos arquivos de 2006 a 2026):

    até 2018                       2019 em diante
    data  "2015-01-01"             "2019/01/01"
    hora  "00:00"                  "0000 UTC"
    falta -9999                    campo vazio

Comum aos dois: 8 linhas de metadados, separador ";", decimal com vírgula,
codificação latin-1, horário em UTC.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LINHAS_DE_METADADOS = 8
COLUNA_CHUVA = 2  # por posição: o nome da coluna muda de acentuação entre anos


def ler_csv(caminho: str | Path) -> pd.Series:
    """Lê um CSV anual e devolve a chuva horária (mm) indexada em UTC."""
    bruto = pd.read_csv(
        caminho,
        sep=";",
        skiprows=LINHAS_DE_METADADOS,
        encoding="latin-1",
        dtype=str,           # nada de adivinhar tipo: convertemos na mão abaixo
        keep_default_na=False,
    )
    data = bruto.iloc[:, 0].str.replace("/", "-", regex=False)
    hora = bruto.iloc[:, 1].str.replace(" UTC", "", regex=False).str.replace(":", "", regex=False)
    instante = pd.to_datetime(data + " " + hora, format="%Y-%m-%d %H%M", utc=True)

    chuva = pd.to_numeric(
        bruto.iloc[:, COLUNA_CHUVA].str.replace(",", ".", regex=False),
        errors="coerce",     # campo vazio vira NaN
    )
    chuva = chuva.where(chuva >= 0)  # -9999 (e qualquer negativo) vira NaN
    return pd.Series(chuva.to_numpy(), index=instante, name="chuva_mm")


def ler_pasta(pasta: str | Path) -> pd.Series:
    """Junta todos os CSVs da pasta numa série horária contínua.

    Horas que não aparecem em arquivo nenhum entram como NaN: a série final
    tem exatamente uma linha por hora entre o primeiro e o último registro.
    Isso importa porque a janela de 3h é calculada por POSIÇÃO; se houvesse
    um buraco no índice, três linhas seguidas poderiam cobrir mais de 3 horas.
    """
    arquivos = sorted(Path(pasta).glob("*.CSV"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum .CSV em {pasta}")
    serie = pd.concat([ler_csv(a) for a in arquivos]).sort_index()
    if serie.index.has_duplicates:
        raise ValueError("hora repetida entre arquivos; conferir os CSVs")
    return serie.asfreq("h")
