"""Finds viable overlaps in the past given a individual (MRN) and date"""

import datetime
import polars as pl
from typing import Iterable
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
        )
    )

    return viable_overlaps

def viable_overlaps(cpe_df: pl.DataFrame, overlaps: pl.DataFrame):
    return {
        entry['sID']: backsearch(entry['sID'], entry['Date'], overlaps)
        for entry in cpe_df.iter_rows(named=True)
    }

def export_viable_overlaps(overlaps: Iterable[pl.DataFrame], path: str=env['outputs']['backsearches']):
    exportable = pl.concat(
        (df
            .with_columns(
                sID=k, 
                cause=(pl.col('patient1') + pl.col('patient2') - k)
            ).select(
                'sID',
                'cause',
                'fID',
                'overlap_start',
                'overlap_end',
                'overlap_duration',
            )
        )
        for k, df in overlaps.items()
    )
    exportable.write_csv(path)

    missings = [k for k,df in overlaps.items() if df.shape[0] == 0]
    missing_df = pl.from_dict(dict(missing=pl.Series(missings)))
    missing_df.write_csv(f"{path}.missing")

def strict_viable_overlaps(cpe_df: pl.DataFrame, viable_overlaps: pl.DataFrame):
    return {
        case: df.filter(
            pl.any(pl.col('patient1', 'patient2').is_in(
                set(cpe_df.select('sID').to_series().to_list()).difference({case})
            ))
        )
        for case, df in viable_overlaps.items()
    }

if __name__ == "__main__":
    # import pickle

    cpe = load_cpe_data()
    cpe = clean_cpe_data(cpe)
    overlaps = load_overlaps()
    viables = viable_overlaps(cpe, overlaps)
    export_viable_overlaps(viables, env['outputs']['backsearches'])
    strict_viables = strict_viable_overlaps(cpe, viables)
    export_viable_overlaps(strict_viables, f"{env['outputs']['backsearches']}.strict")

    # with open("viable_overlaps.pkl", 'wb') as fp:
        # pickle.dump(viables, fp)
