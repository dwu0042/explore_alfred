"""Aggregates simulation infections by location (ward) and outputs
the mean and variance of the proportion of infections thru that loc
(kinda jank, since we cannot ensure that the sum(means) = 1)
"""

import polars as pl
import glob

def load(path_template:str):
    return {
        path: pl.read_csv(path) for path in glob.glob(path_template)
    }

def ward_stats(dfs):

    frequencies = {path: df.groupby('location').count().sort('count').filter(pl.col('location').is_not_null()) 
                   for path, df in dfs.items()}

    norm_freqs = [df.select(['location', pl.col('count')/pl.col('count').sum()])
                    .rename({'count': f"count_{path.split('.')[1]}"})
                  for path, df in frequencies.items()]

    stats = norm_freqs[0]
    for df in norm_freqs[1:]:
        stats.join(df, on='location', how='outer')

    perc_inf = (stats.filter(pl.col('location').is_not_null())
                     .fill_null(0)
                     .with_columns(pl.concat_list(pl.all().exclude('location')).alias('clist'))
                     .select([
                        'location', 
                        pl.col('clist').arr.eval(pl.element().mean(), parallel=True).arr.first().alias('mean_inf'), 
                        pl.col('clist').arr.eval(pl.element().var(), parallel=True).arr.first().alias('var_inf')
                      ])
    )
    return perc_inf

if __name__ == "__main__":
    import sys
    template = sys.argv[1]
    dfs = load(template)
    stats = ward_stats(dfs)
    stats.write_csv(sys.argv[2])