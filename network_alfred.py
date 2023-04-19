import overlap_residence as ovr
import polars as pl

from env_reader import env

def read(path=env['outputs']['clean_data']):
    df = pl.read_csv(path, try_parse_dates=True)
    return df

def overlap_dict(df:pl.DataFrame):
    return ovr.compute_shared_residence_times(df.sort('Adate').iter_rows())

def build_overlaps_df(ov_dict):
    overlap_rows = [
        (this, other, location, duration.total_seconds()/60/60/24, start, end)
        for this, this_info in ov_dict.items()
        for location, loc_info in this_info.items()
        for other, (duration, (start, end)) in loc_info.items()
    ]
    overlap_df = pl.from_records(overlap_rows, schema={
        'patient1': pl.UInt32,
        'patient2': pl.UInt32,
        'ward': pl.Utf8,
        'duration': pl.Float64,
        'overlap_start': None,
        'overlap_end': None,
    })
    return overlap_df

def export_overlaps(ov_df: pl.DataFrame, path:str):
    ov_df.write_csv(path)

def build_network(overlap_dict):
    # overlap_dict = ovr.compute_shared_residence_times(df.iter_rows())
    G = ovr.make_overlap_graph(overlap_dict)
    for nd, info in G.nodes(data=True):
        if 'duration' in info:
            G.nodes[nd]['duration'] = info['duration'].seconds/60/60  # -> hours
    return G
    
def export_graph(G, export_name=env['outputs']['network']):
    ovr.export_gml(G, f"{export_name}.gml")
    ovr.export_graphviz(G, f"{export_name}.dot")

if __name__ == "__main__":
    df = read()
    ovl = overlap_dict(df)
    ov_df = build_overlaps_df(ovl)
    ov_df.sort('overlap_start').write_csv(env['output']['overlaps'])
    # G = build_network(ovl)
    # export_graph(G)