import datetime
import polars as pl
from env_reader import env

_ref_date = datetime.datetime(year=2013, month=4, day=1)

def load(path=env['outputs']['clean_data']):
    return pl.read_csv(path, try_parse_dates=True)

def lookup(record:pl.DataFrame, col:str):
    return record.select(col).item()

def compute_single_overlap(record: pl.DataFrame, df: pl.DataFrame):
    not_same = pl.col('sID') != record['sID']

    same_facility = pl.col('fID') == record['fID']
    
    adate_inside = (
        (record['Adate'] < pl.col('Adate'))
        & (pl.col('Adate') < record['Ddate'])
    )

    ddate_inside = (
        (record['Adate'] < pl.col('Ddate'))
        & (pl.col('Ddate') < record['Ddate'])
    )

    overlaps = df.filter(not_same & same_facility & (adate_inside | ddate_inside))

    return overlaps.with_columns(
        overlap_start= pl.max(pl.col('Adate'), record['Adate']),
        overlap_end= pl.min(pl.col('Ddate'), record['Ddate'])
    ).with_columns(
        overlap_duration= (pl.col('overlap_end') - pl.col('overlap_start')).dt.seconds()/60/60/24
    ).with_columns(
        patient1= record['sID']
    ).select([
        'patient1',
        pl.col('sID').alias('patient2'),
        'fID',
        'overlap_start',
        'overlap_end',
        'overlap_duration'
    ])

def dump_csv_fragment(df: pl.DataFrame, path: str, header=False):
    with open(path, 'ab') as fp:
        df.write_csv(fp, has_header=header)

def compute_and_write_overlaps(df:pl.DataFrame, path:str=env['outputs']['overlaps'], print_freq=None, allow_duplicates=True):
    L = df.shape[0]
    for i, row in enumerate(df.iter_rows(named=True)):
        overlap = compute_single_overlap(row, df)
        dump_csv_fragment(overlap, path, header=(i==0))
        if print_freq and i%print_freq == 0:
            print("Processed row", i, "of", L-1)

    if not allow_duplicates:
        clean_duplicates(path)

def clean_duplicates(path):
    overlaps_df = pl.scan_csv(path, try_parse_dates=True)
    overlaps_df = (
        overlaps_df
        .with_columns(patients=pl.concat_list(pl.col('patient1', 'patient2')).arr.sort())
        .with_columns(
            patient1 = pl.col('patients').arr.get(0),
            patient2 = pl.col('patients').arr.get(1),
        )
        .select(pl.exclude('patients'))
        .unique()
    )
    overlaps_df.collect().write_csv(path)

if __name__ == "__main__":
    df = load()
    compute_and_write_overlaps(df, print_freq=int((df.shape[0])/10), allow_duplicates=False)