"""Finds viable overlaps in the past given a individual (MRN) and date"""

import datetime
import polars as pl
from env_reader import env

_end_of_period = datetime.date(year=2019, month=4, day=30)

def load_cpe_data(path: str=env['data']['cpe']):
    return pl.read_csv(path, try_parse_dates=True)

def clean_cpe_data(df: pl.DataFrame, target="E. cloacae"):
    df = (df
        .filter(pl.col('organism') == target)
        .filter(pl.col('Dates positive') <= _end_of_period)
    )
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

def viable_overlaps(cpe_df: pl.DataFrame, overlaps: pl.DataFrame):
    return {
        entry['sID']: backsearch(entry['sID'], entry['Date'], overlaps)
        for entry in cpe_df.iter_rows(named=True)
    }

if __name__ == "__main__":
    import pickle

    cpe = load_cpe_data()
    cpe = clean_cpe_data(cpe)
    overlaps = load_overlaps()
    viables = viable_overlaps(cpe, overlaps)
    with open("viable_overlaps.pkl", 'wb') as fp:
        pickle.dump(viables, fp)
