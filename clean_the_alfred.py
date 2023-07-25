import transfer_linkage as tl
import polars as pl
from matplotlib import pyplot as plt
import datetime

import drop_alfred_wards as drop

_dtfmt = '%d/%m/%Y %H:%M'

data_collection_end = {
    1: datetime.datetime(year=2014, month=4, day=1, hour=0, minute=0),
    2: datetime.datetime(year=2019, month=5, day=1, hour=0, minute=0),
}

def read_df(path:str) -> pl.DataFrame:
    df = tl.cleaner.ingest_csv(path)

    return df

def preprocess(df:pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        WardCode=pl.col('Ward').str.split(' - ').list.get(0),
    )
    # small preprocessing for next step: we need sortable dates
    df = tl.cleaner.standardise_column_names(df, subject_id='MRN', 
                                             facility_id='WardCode',
                                             admission_date='WardAdmission',
                                             discharge_date='WardDischarge')
    df = tl.cleaner.coerce_data_types(df, convert_dates=True, date_format=_dtfmt)
    df = df.sort('sID', 'Adate')

    # use backwards filling on each separate chunk (by patient) to fill in gaps of the records
    df = pl.concat([dfi.fill_null(strategy='backward') for dfi in df.partition_by('sID')])

    # drop wards that are not required
    df = df.filter(pl.col('fID').is_in(drop.drop_ward_codes).is_not())

    fill_val = data_collection_end[df.select('time_period')[0,].item()]
    # search for null discharge dates
    print(df.filter(pl.any_horizontal(pl.col('Ddate').is_null())).shape[0], 'null values exist')
    (df.filter(pl.any_horizontal(pl.col('Ddate').is_null()))
        .select('sID', 'fID', 'Adate', 'Ddate')
        .with_columns(Duration=(fill_val-pl.col('Adate')).dt.seconds()/60/60/24)
        .write_csv(f"nulls_{df.select('time_period')[0,].item()}.csv")
    )
    # df.filter(pl.any(pl.col('Ddate').is_null())).write_csv(f"nulls_{df.select('time_period')[0,].item()}.csv")


    # catch any end problem nulls by filling with the known end time of the period
    df = df.fill_null(fill_val)

    return df

def load_and_preclean(path:str) -> pl.DataFrame:
    df = read_df(path)
    parted_dfs = df.partition_by('time_period')
    return pl.concat([preprocess(dfxi) for dfxi in parted_dfs])

def yield_clean_df(original_df:pl.DataFrame) -> pl.DataFrame:
    return tl.cleaner.clean_database(original_df, convert_dates=False,
                                     retain_auxiliary_data=False)

if __name__ == "__main__":
    from env_reader import env

    df = load_and_preclean(env['data']['raw'])
    df_clean = yield_clean_df(df)

    df_clean.write_csv(env['outputs']['clean_data'])