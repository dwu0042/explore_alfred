from matplotlib import pyplot as plt
import polars as pl

def load_sim(path: str):
    return pl.read_csv(path)

def plot_sim_infections(sim: pl.DataFrame):
    f = plt.figure()
    ax = f.add_subplot()
    ax.plot(sim.select('time').to_series(), sim.select('infected').to_series())
    ax.set_xlabel('Time since 1 April 2013 (days)')
    ax.set_ylabel("Cumulative Infections")

    return f, ax

if __name__ == "__main__":
    import sys

    sim = load_sim(sys.argv[1])
    plot_sim_infections(sim)

    plt.show(block=False)
    input()
