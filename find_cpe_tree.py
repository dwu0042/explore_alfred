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
        pl.col('st').alias('strain')
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
    # TODO: keys need to be unique, sID is not unique
    # return {
    #     entry['sID']: backsearch(entry['sID'], entry['Date'], overlaps)
    #     for entry in cpe_df.iter_rows(named=True)
    # }
    return (pl.concat(
        backsearch(entry['sID'], entry['Date'], overlaps)
        .with_columns(**{k: pl.lit(v, dtype=d) for (k,v),d in zip(entry.items(), cpe_df.dtypes)})
        for entry in cpe_df.iter_rows(named="True")
    )
    .with_columns(cause=pl.col('patient1')+pl.col('patient2')-pl.col('sID'))
    .select('sID', 'cause', 'fID', 'overlap_start', 'overlap_end', 'overlap_duration', 
            pl.col('Date').alias('date_positive'), pl.col('Reported').alias('date_reported'),
            'strain')
    )

def export_viable_overlaps(overlaps: Iterable[pl.DataFrame], path: str=env['outputs']['backsearches'], missing_from: pl.Series|list=None):
    overlaps.write_csv(path)

    if missing_from is not None:
        missings = set(missing_from).difference(set(overlaps.select('sID').unique().to_series()))
        missing_df = pl.from_dict(dict(missing=pl.Series(list(missings))))
        missing_df.write_csv(f"{path}.missing")

def strict_viable_overlaps(cpe_df: pl.DataFrame, viable_overlaps: pl.DataFrame):
    # make assumption that strains do not mutate over time, 
    # i.e. infections only lead to same strain
    # we can use this to detect structural holes
    # esp if we compare with the slightly less strict version

    generic_st = ['unknown', 'Novel ST', 'ST Novel']

    return (
        viable_overlaps.join(cpe_df, left_on='cause', right_on='sID', how='left', suffix='_cause')
        .filter(
            (pl.col('cause').is_in(cpe_df.select('sID').unique().to_series()))
            & (
                (pl.col('strain') == pl.col('strain_cause'))
                | (pl.col('strain').is_in(generic_st))
            )
        )
        .select('sID', 'cause', 'fID', 'overlap_start', 'overlap_end', 'overlap_duration', 
                'date_positive', 'date_reported', 'strain')
    )

if __name__ == "__main__":
    # import pickle

    cpe = load_cpe_data()
    cpe = clean_cpe_data(cpe)
    overlaps = load_overlaps()
    viables = viable_overlaps(cpe, overlaps)
    export_viable_overlaps(viables, missing_from=cpe.select('sID').to_series())
    strict_viables = strict_viable_overlaps(cpe, viables)
    export_viable_overlaps(strict_viables, path=f"{env['outputs']['backsearches']}.strict", 
                           missing_from=cpe.select('sID').to_series())

    # with open("viable_overlaps.pkl", 'wb') as fp:
        # pickle.dump(viables, fp)
