"""Infections transmitted through contaminated locations

States:
Individuals:
    S: Susceptible
    I: Infected

Reactions:
Infect: S -> I : 
"""

import math
import random
from enum import auto
from typing import Hashable, Iterable, Mapping
import dataclasses
import pathlib
import datetime
import polars as pl
import yaml
import pickle

from mkmover import abm
from alfred_model_framework import Model, arg_handler


def logistic(x):
    return (1 - math.exp(-x)) / (1 + math.exp(-x))


class Location(abm.Location):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_contaminated = 0
        self.impulse = 0
        self.decay_rate = 0
        self.baseline = 0

    @property
    def effective_clean_time(self):
        threshold = 0.01
        if (threshold - self.baseline < 0) or (self.impulse == 0):
            return float("Inf")
        return self.last_contaminated + self.decay_rate * math.log(
            self.impulse / (threshold - self.baseline)
        )

    def add_impulse(self, t: float, impulse: float):
        self.impulse = self.current_level(t) + impulse
        self.last_contaminated = t

    def current_level(self, t: float):
        return (
            math.exp(-(t - self.last_contaminated) / self.decay_rate) * self.impulse
            + self.baseline
        )

    def odds(self, t: float):
        return logistic(self.current_level(t))


class AlfredEvent(abm.EventType):
    Move = auto()
    Infect = auto()
    Contaminate = auto()


@dataclasses.dataclass(frozen=True, eq=True, order=True)
class Event(abm.Event):
    aux: str | None = None


class Model(Model):
    _ref_date = datetime.datetime(year=2013, month=4, day=1)
    _NULL_LOC = Location("None")


    def __init__(self, event_file: str | pathlib.Path, parameters: Mapping[str, float]):
        super().__init__()
        self.schema = {
            "time": pl.Float64,
            "infected": pl.Int64,
            "agent": pl.Int64,
            "location": pl.Utf8,
        }
        if event_file is not None:
            self.transition_events = pl.read_csv(event_file, try_parse_dates=True)
        else:
            self.transition_events = pl.DataFrame(
                schema={
                    "sID": pl.Int64,
                    "fID": pl.Utf8,
                    "Date": pl.Datetime,
                }
            )
        self.state.add_location(self._NULL_LOC)
        self.update_parameters(parameters)
        self.state._event_base = Event
        self._rejects = 0

    def update_parameters(self, parameters):
        self.base_infect_rate = parameters.get("infection", 0)
        self.impulse_value = parameters.get("impulse", 1)
        self.impulse_delay = parameters.get("impulse_delay", 0)
        self.impulse_decay = parameters.get("impulse_decay", 1)
        self.icu_baseline = parameters.get("icu_baseline", 0.5)

    def initialise_state(self):
        """Populate individuals and locations"""

        locations = self.transition_events.select("fID").unique().to_series()
        for location in locations:
            if location is None:
                # already dealt with
                continue
            location = Location(location)
            location.decay_rate = self.impulse_decay
            self.state.add_location(location)

        agents = self.transition_events.select("sID").unique().to_series()
        for agent in agents:
            agent = abm.Agent(name=agent)
            agent.infected = "S"
            agent.move_to(self._NULL_LOC.id)
            self.state.add_agent(agent)

    def generate_initial_events(self, permute=None):
        sim_t = (
            (pl.col("Date") - self._ref_date).dt.seconds() / 60 / 60 / 24
        )  # base unit days
        events = self.transition_events.with_columns(sim_t.alias("simt"))
        for chunk in events.partition_by("sID"):
            if permute:
                # roll a permutation value
                # in range -p to p
                # where it is beta-distribed from [-p, p]
                dt = permute * (2 * random.betavariate(2, 2) - 1)
            else:
                dt = 0
            for event in chunk.iter_rows(named=True):
                self.state.add_event(
                    t=max(0, event["simt"] + dt),  # just clipping at zero here
                    event_type=AlfredEvent.Move,
                    agent=event["sID"],
                    aux=str(event["fID"]),
                )

    def do_infect(self, location: Hashable):
        """Infect an individual at the location (maybe)"""

        loc_obj = self.state.locations[location]

        # schedule next infection event at location
        # if the next infection event will happen where the contamination is below a threshold,
        # just ignore it, since the location is effectively clean
        # this acts as a clearance time
        next_time = self.state.t + random.expovariate(self.base_infect_rate)
        if next_time < loc_obj.effective_clean_time:
            self.state.add_event(
                t=next_time,
                event_type=AlfredEvent.Infect,
                agent=location,
            )

        # sanity check rejects
        targets = loc_obj.occupants
        if len(targets) < 1:
            # no one to infect
            return []
        if location == "None":
            # None is home
            return []
        if loc_obj is self._NULL_LOC:
            # This is also home by construction
            return []

        # check for odds to accept
        roll = random.random()
        if roll > loc_obj.odds(self.state.t):
            self._rejects += 1
            # reject
            return []

        # roll for a random individual to infect
        cand = random.choice(list(targets))
        cand_obj = self.state.agents[cand]
        if cand_obj.state == "S":
            # infect the susceptible
            cand_obj.infected = "I"
            # queue a burst of infection impulse
            impulse_delay = (
                random.expovariate(self.impulse_delay) if self.impulse_delay else 0
            )
            self.state.add_event(
                t=(self.state.t + impulse_delay),
                event_type=AlfredEvent.Contaminate,
                agent=cand,
            )

            return [(cand, location, 1)]
        return []

    def do_contaminate(self, agent: Hashable):
        """Add an impulse of contamination to a location"""

        agent_obj = self.state.agents[agent]
        location = agent_obj.location
        loc_obj = self.state.locations[location]

        # schedule the next contamination event
        # only if the agent is going to be not home in the future
        if len(self.state.event_map[agent]) > 0:
            impulse_delay = (
                random.expovariate(self.impulse_delay) if self.impulse_delay else 0
            )
            self.state.add_event(
                t=(self.state.t + impulse_delay),
                event_type=AlfredEvent.Contaminate,
                agent=agent,
            )

        if loc_obj.id == self._NULL_LOC.id:
            # don't do anything at home
            return []

        # check if the location _was_ clean
        was_clean = loc_obj.effective_clean_time < self.state.t

        # add the impulse
        loc_obj.add_impulse(self.state.t, self.impulse_value)

        # schedule next infection event if was clean
        if was_clean:
            self.state.add_event(
                t=(self.state.t + random.expovariate(self.base_infect_rate)),
                event_type=AlfredEvent.Infect,
                agent=location,
            )

        return [(agent, location, 0)]

    def do_next_move(self, agent: Hashable, to: Hashable):
        self.state.move(agent, location=to)
        return []

    def handle_next_event(self, event):
        match event.event_type:
            case AlfredEvent.Move:
                return self.do_next_move(event.agent, event.aux)
            case AlfredEvent.Infect:
                return self.do_infect(event.agent)
            case AlfredEvent.Contaminate:
                return self.do_contaminate(event.agent)
            case _:
                return []

    def seed(self):
        _ICU = "A-ICU"
        # we contaminate the ICU with a baseline value
        self.state.locations[_ICU].baseline = self.icu_baseline
        # and schedule a first infection
        self.state.add_event(
            t=(self.state.t + random.expovariate(self.base_infect_rate)),
            event_type=AlfredEvent.Infect,
            agent=_ICU,
        )
        self.history.append([0, 0, None, "A-ICU"])

    def get_status_str(self):
        n_inf = len(
            [agents for agents, info in self.state.agents.items() if info.state == "I"]
        )
        return f"{n_inf} infected"

    def update_history(self, records: Iterable):
        for agent, location, change in records:
            self.history.append(
                [self.state.t, self.history[-1][1] + change, agent, location]
            )

    def write_metadata(self, path: str, opts):
        with open(path, "a") as fp:
            fp.write(
                yaml.safe_dump(
                    [
                        {
                            "result": str(opts.output),
                            "parameters": {
                                "infection": self.base_infect_rate,
                                "impulse": self.impulse_value,
                                "impulse_delay": self.impulse_delay,
                                "impulse_decay": self.impulse_decay,
                                "icu_baseline": self.icu_baseline,
                            },
                            "maxtime": opts.maxtime,
                            "exectime": datetime.datetime.now(),
                            "permute_value": opts.permute_value
                            if not opts.model
                            else None,
                            "base_model": str(opts.model)
                            if opts.model
                            else str(opts.dump_file)
                            if opts.dump_model
                            else None,
                        }
                    ]
                )
            )


def parse_args():
    return arg_handler(
        [
            dict(
                names=["--infection", "-i"],
                type=float,
                default=0,
            ),
            dict(
                names=["--impulse", "-iv"],
                type=float,
                default=1,
            ),
            dict(
                names=["--impulse-delay", "-id"],
                type=float,
                default=0,
            ),
            dict(
                names=["--impulse_decay", "-dc"],
                type=float,
                default=1,
            ),
            dict(
                names=["--icu_baseline", "-icu"],
                type=float,
                default=0.5,
            ),
            dict(
                names=["--events", "-e"],
                type=pathlib.Path,
            ),
            dict(
                names=["--permute-value", "-z"],
                type=float,
                default=0,
            ),
        ]
    )


def main():
    opts = parse_args()

    model_parameters = {
        attr: getattr(opts, attr)
        for attr in [
            "infection",
            "impulse",
            "impulse_delay",
            "impulse_decay",
            "icu_baseline",
        ]
    }

    if opts.model is None:
        print("Building model...")
        sim = Model(event_file=opts.events, parameters=model_parameters)
        print("Model built")
        print("Initialising agents...")
        sim.initialise_state()
        print("Agents initialised")
        print("Generating movement events...")
        sim.generate_initial_events(permute=opts.permute_value)
        print("Movement events genreated")

        if opts.dump_model:
            print(f"Dumping model to {opts.dump_file} ...")
            with open(opts.dump_file, "wb") as fp:
                pickle.dump(sim, fp)
            print("Model dumped")

    else:
        print(f"Reading model from {opts.model} ...")
        with open(opts.model, "rb") as fp:
            sim = pickle.load(fp)
        print("Model loaded")

    print(f"N={len(sim.state.agents)} agents")
    sim.update_parameters(model_parameters)
    print("Setting the initial conditions...")
    sim.seed()
    print("Initial conditions set")
    sim.simulate(until=opts.maxtime, print_freq=opts.print_freq)
    print("Simulation finished")
    print(f"{sim._rejects} rejections from nonMarkov impulse")
    print("Recording results...")
    actual_outpath = sim.write_result(opts.output)
    opts.output = actual_outpath
    print(f"Recorded to {opts.output}")
    print(f"Writing metadata to {opts.metadata} ...")
    sim.write_metadata(opts.metadata, opts=opts)
    print("Metadata recorded.")
    print("Done.")


if __name__ == "__main__":
    main()
