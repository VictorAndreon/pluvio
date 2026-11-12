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


def ler_metadados(caminho: str | Path) -> dict:
    """Código, nome e coordenadas da estação, tirados das 8 linhas do cabeçalho.

    As chaves do cabeçalho mudam de acentuação entre anos ("ESTAÇÃO", "ESTACAO",
    "ESTAC?O"), então lemos por POSIÇÃO da linha, que é estável.
    """
    with open(caminho, encoding="latin-1") as arquivo:
        valores = [arquivo.readline().rstrip("\n").split(";", 1)[1].strip() for _ in range(LINHAS_DE_METADADOS)]
    return {
        "nome": valores[2],
        "codigo": valores[3],
        "latitude": float(valores[4].replace(",", ".")),
        "longitude": float(valores[5].replace(",", ".")),
        "altitude_m": float(valores[6].replace(",", ".")),
    }


def listar_estacoes(pasta: str | Path) -> dict[str, dict]:
    """{código: metadados} de todas as estações com CSV na pasta.

    Se a coordenada mudou ao longo dos anos (o INMET corrigiu a de Passo Fundo
    em 2019), vale a do arquivo mais recente.
    """
    estacoes: dict[str, dict] = {}
    for arquivo in sorted(Path(pasta).glob("*.CSV"), key=_ano_do_arquivo):
        meta = ler_metadados(arquivo)
        estacoes[meta["codigo"]] = meta
    return estacoes


def _ano_do_arquivo(arquivo: Path) -> str:
    return arquivo.stem[-4:]      # o nome termina em "..._A_31-12-2024"


def ler_pasta(pasta: str | Path, codigo: str = "A839") -> pd.Series:
    """Junta todos os CSVs de UMA estação numa série horária contínua.

    Horas que não aparecem em arquivo nenhum entram como NaN: a série final
    tem exatamente uma linha por hora entre o primeiro e o último registro.
    Isso importa porque a janela de 3h é calculada por POSIÇÃO; se houvesse
    um buraco no índice, três linhas seguidas poderiam cobrir mais de 3 horas.
    """
    arquivos = [a for a in sorted(Path(pasta).glob("*.CSV")) if ler_metadados(a)["codigo"] == codigo]
    if not arquivos:
        raise FileNotFoundError(f"nenhum .CSV da estação {codigo} em {pasta}")
    serie = pd.concat([ler_csv(a) for a in arquivos]).sort_index()
    if serie.index.has_duplicates:
        raise ValueError(f"{codigo}: hora repetida entre arquivos; conferir os CSVs")
    return serie.asfreq("h")
