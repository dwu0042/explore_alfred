import polars as pl
from transfer_linkage import parse_transitions as pt

if __name__ == "__main__":
    from env_reader import env

    df = pl.read_csv(env['outputs']['clean_data'], try_parse_dates=True)
    transitions = pt.parse_transitions(df)
    transitions.write_csv(env['outputs']['transitions'])