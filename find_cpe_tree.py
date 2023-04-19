"""Finds viable overlaps in the past given a individual (MRN) and date"""

import polars as pl
from env_reader import env

def load_cpe_data(path: str):
    return pl.read_csv(path, try_parse_dates=True)

def clean_cpe_data(df: pl.DataFrame, target="E. cloacae"):
    df = df.filter(pl.col('organism') == target)
    return df.select([
        pl.col('mrn').alias('sID'), 
        pl.col('Dates positive').alias('Date'),
        pl.col('Date reported').alias('Reported'),
    ])

def load_overlaps(path: str=env['outputs']['overlaps']):
    return pl.read_csv(path, try_parse_dates=True)

def backsearch(individual, timing, overlaps):
    viable_overlaps = overlaps.filter(
        pl.any(
            pl.col('patient1', 'patient2') == individual
        ) & (
            pl.col('overlap_start') < timing
        ) & (
            timing < pl.col('overlap_end')
        )
    )

    return viable_overlaps