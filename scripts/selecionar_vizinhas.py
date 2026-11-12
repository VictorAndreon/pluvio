"""Copia para data/raw/inmet os CSVs das estações do INMET próximas de Passo Fundo.

Os .zip anuais do INMET trazem todas as estações do Brasil. Extraia os anos
de 2024 a 2026 em alguma pasta e rode, a partir da raiz do repositório:

    python scripts/selecionar_vizinhas.py CAMINHO/DA/PASTA/EXTRAIDA

Por padrão pega as estações a até 150 km. O script só lista e copia; não
altera nenhum arquivo de origem.
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from pluvio.inmet import ler_metadados

PASSO_FUNDO = (-28.2267, -52.4036)
ANOS = ("2024", "2025", "2026")


def distancia_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distância entre dois pontos (lat, lon) pela fórmula de haversine."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("origem", help="pasta com os CSVs extraídos dos .zip do INMET (procura em subpastas)")
    ap.add_argument("--raio-km", type=float, default=150.0)
    ap.add_argument("--destino", default="data/raw/inmet")
    ap.add_argument("--so-listar", action="store_true", help="mostra o que copiaria, sem copiar")
    args = ap.parse_args()

    achados = []
    for arquivo in sorted(Path(args.origem).rglob("*.CSV")):
        if not arquivo.stem.endswith(ANOS):
            continue
        meta = ler_metadados(arquivo)
        km = distancia_km(PASSO_FUNDO, (meta["latitude"], meta["longitude"]))
        if km <= args.raio_km:
            achados.append((km, meta["codigo"], meta["nome"], arquivo))

    destino = Path(args.destino)
    for km, codigo, nome, arquivo in sorted(achados, key=lambda x: (x[0], x[3].name)):
        acao = "listado"
        if not args.so_listar:
            alvo = destino / arquivo.name
            if alvo.exists():
                acao = "já existe"
            else:
                shutil.copy2(arquivo, alvo)
                acao = "copiado"
        print(f"{km:6.1f} km  {codigo}  {nome:<28} {arquivo.name}  [{acao}]")
    print(f"\n{len({c for _, c, _, _ in achados})} estações, {len(achados)} arquivos dentro de {args.raio_km:.0f} km")


if __name__ == "__main__":
    main()
