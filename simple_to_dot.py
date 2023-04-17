import polars as pl

def to_dot(path: str):
    df = pl.read_csv(path)
    nodes = df.select(pl.format('{};\n', pl.col('patient'))).unique().to_series().to_list()
    edges = df.select(pl.format('{} -> {};\n', pl.col('cause').fill_null('"X"'), pl.col('patient'))).unique().to_series().to_list()

    with open(f"{path}.dot", 'w') as fp:
        fp.write("strict digraph {\n")
        fp.write("\"X\";\n")
        fp.writelines(nodes)
        fp.writelines(edges)
        fp.write('}')

if __name__ == "__main__":
    import sys
    to_dot(sys.argv[1])