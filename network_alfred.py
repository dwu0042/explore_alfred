import overlap_residence as ovr
import polars as pl

from env_reader import env

def read(path=env['outputs']['clean_data']):
    df = pl.read_csv(path, try_parse_dates=True)
    return df

def build_network(df:pl.DataFrame):
    overlap_dict = ovr.compute_shared_residence_times(df.iter_rows())
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
    G = build_network(df)
    export_graph(G)