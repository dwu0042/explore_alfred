
import pathlib
import dataclasses
import polars as pl
from typing import Iterable, Hashable
from enum import Enum
import datetime
import random

from mkmover import abm

@dataclasses.dataclass(frozen=True, eq=True, order=True)
class Event(abm.Event):
    aux: str | None = None

AlfredEventType = Enum('AlfredEventType', ['Move', 'Infect'], type=abm.EventType)

class Model(abm.Model):
    _ref_date = datetime.datetime(year=2013, month=4, day=1)
    _NULL_LOC = abm.Location('None')

    def __init__(self, event_file_path: str, infection_rate: float):
        super().__init__()
        if event_file_path is not None:
            self.transition_events = pl.read_csv(event_file_path, try_parse_dates=True)
        else:
            self.transition_events = pl.DataFrame(schema={
                'sID': pl.Float64,
                'fID': pl.Utf8,
                'Date': pl.Datetime,
            })
        self.state.add_location(self._NULL_LOC)
        self.infection_rate = infection_rate
        self.state._event_base = Event

    def initialise_state(self):
        """Populate the individuals, set all individuals to null location, create wards."""

        locations = self.transition_events.select('fID').unique().to_series()
        for location in locations:
            if location is None:
                # already dealt with
                continue
            location = abm.Location(location)
            self.state.add_location(location)

        agents = self.transition_events.select('sID').unique().to_series()
        for agent in agents:
            agent = abm.Agent(name = agent)
            agent.infected = 'S'
            agent.move_to(self._NULL_LOC.id)
            self.state.add_agent(agent)
            # self.state.move(agent.id, self._NULL_LOC.id)

    def generate_initial_events(self, permute=None):
        sim_t = (pl.col('Date') - self._ref_date).dt.seconds() / 60 / 60 / 24 # base unit days
        events = self.transition_events.with_columns(sim_t.alias('simt'))
        for chunk in events.partition_by('sID'):
            if permute:
                # roll a permutation value
                # in range -p to p
                # where it is beta-distribed from [-p, p]
                dt = permute * (2*random.betavariate(2, 2) - 1)  
            for event in chunk.iter_rows(named=True):
                self.state.add_event(
                    t=max(0, event['simt']+dt), # just clipping at zero here
                    event_type=AlfredEventType.Move, 
                    agent=event['sID'],
                    aux=str(event['fID']),
                )

    def update_agent_events(self, agent: Hashable, occurred_event_type: AlfredEventType):
        # we deal with this in the event handling any way
        pass

    def handle_next_event(self):
        next_event = self.state.event_queue.pop()
        self.state.t = next_event.t
        records = []
        if next_event.event_type is AlfredEventType.Move:
            records = self.do_next_move(agent=next_event.agent, to=next_event.aux)
        elif next_event.event_type is AlfredEventType.Infect:
            records = self.do_potential_infect(next_event.agent)
        else:
            raise ValueError(f"{next_event.event_type} is not a known Event type")

        self.state.event_map[next_event.agent].discard(next_event)
        return records

    def schedule_next_infection_event(self, agent: Hashable):
        self.state.add_event(
            t=(self.state.t + random.expovariate(self.infection_rate)),
            event_type=AlfredEventType.Infect,
            agent=agent
        )

    def do_next_move(self, agent: Hashable, to: Hashable):
        self.state.move(agent, location=to)
        return []

    def do_potential_infect(self, agent: Hashable):
        # find infector object and location
        agent_obj = self.state.agents[agent]
        location = agent_obj.location
        loc_obj = self.state.locations[location]
        targets = loc_obj.occupants

        # queue next infection event for the current infector
        # only if there is more in the event queue than this single infection event
        # we know all individuals return to null
        if len(self.state.event_map[agent]) > 1:
            self.schedule_next_infection_event(agent)

        if len(targets) < 2:
            # no one to infect
            return []
        if location == 'None':
            return []
        if loc_obj is self._NULL_LOC:
            # at home
            return []
        
        # find a target to infect that is not the agent itself
        cands = random.sample(list(targets), k=2)
        if cands[0] == agent:
            cands.remove(agent)
        cand_obj = self.state.agents[cands[0]]
        if cand_obj.infected == 'S':
            cand_obj.infected = 'I'
            # queue infection events for this new infector
            self.schedule_next_infection_event(cand_obj.id)
            return [(agent, cand_obj.id, location)]
        return []

    def seed(self, fixed_seed=None, seed_strategy='fID'):
        # find a person that does something in the first N events
        # infect them at t=0
        if fixed_seed is None:
            if seed_strategy == 'fID':
                index = random.randint(0, 100)
                event = self.transition_events.filter(pl.col('fID') == 'A-ICU').sort('Date')[index]
                agent = event.select('sID').item()
            else:
                index = random.randint(0, 100)
                event = self.state.event_queue[index]
                agent = event.agent
        else:
            agent = fixed_seed
        self._seed = agent
        self.state.agents[agent].infected = 'I'
        print(f"Agent: {agent} infected.")
        print("Their patient journey:")
        print(self.transition_events.filter(pl.col('sID') == agent).with_columns((pl.col('Date') - self._ref_date).dt.days().alias('simt')))
        # they start blasting at time 0
        self.state.add_event(
            t=0,
            event_type=AlfredEventType.Infect,
            agent=agent,
        )
        self.history.append([0, 1, agent, None, None])

    def simulate(self, until=None, print_freq=10):
        rec = 0
        if until is None:
            until = float('Inf')
        rough_time = time.perf_counter()
        # loop until either we hit time limit OR we run out of events
        while self.state.t < until and self.state.event_queue:
            records = self.handle_next_event()
            if print_freq and (self.state.t > (rec+1)*print_freq):
                n_inf = len([agents for agents, info in self.state.agents.items() if info.state == 'I'])
                curr_time = time.perf_counter()
                print(f"t={(rec+1)*print_freq}, {n_inf} infected, walltime={curr_time-rough_time:.0}s")
                rough_time = curr_time
                rec += 1
            for cause, changed, loc in records:
                self.history.append([self.state.t, self.history[-1][1] + 1, changed, cause, loc])

    def write_result(self, path: pathlib.Path):
        df = pl.from_records(self.history, schema={
            'time': pl.Float64, 
            'infected': pl.Int64,
            'patient': pl.Int64,
            'cause': pl.Int64,
            'location': pl.Utf8,
            }
        )
        outpath = path.with_stem(f"{path.stem}_{int(time.time())}")
        df.write_csv(outpath)
        return outpath

    def write_metadata(self, path: str, opts: str):
        with open(path, 'a') as fp:
            fp.write(yaml.safe_dump([{
                'result': str(opts.output),
                'infection_rate': self.infection_rate,
                'maxtime': opts.maxtime,
                'seed': self._seed,
                'exectime': datetime.datetime.now(),
                'permute_value': opts.permute_value if not opts.model else None,
                'base_model': str(opts.model) if opts.model else str(opts.dump_file) if opts.dump_model else None,
            }]))

if __name__ == "__main__":
    import pickle, time
    import argparse
    import yaml
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', action='store', default=None)

    parser.add_argument('--events', '-e', action='store', 
                        default=None)
    parser.add_argument('--infection', '-i', action='store', type=float,
                        default=None)
    parser.add_argument('--model', '-m', action='store', type=pathlib.Path,
                        default=None, 
                        help="Loads in existing base model. Overrides and ignores 'events', 'permute-value', 'dump-model', 'dump-file'")
    parser.add_argument('--seed', '-s', action='store', type=int,
                        default=None)
    parser.add_argument('--permute-value', '-z', action='store', type=float,
                        default=None)
    parser.add_argument('--maxtime', '-t', action='store', type=float,
                        default=None)
    parser.add_argument('--print-freq', '-p', action='store', type=int,
                        default=0)
    parser.add_argument('--output', '-o', action='store', type=pathlib.Path,
                        default=f"sim_result.{int(time.time())}.csv")
    parser.add_argument('--dump-model', '-d', action='store_true',
                        default=False)
    parser.add_argument('--dump-file', '-q', action='store', type=pathlib.Path,
                        default="movement_model.pkl")
    parser.add_argument('--metadata', '-w', action='store', type=pathlib.Path,
                        default="metadata.yaml")
    parser.add_argument('--dry', '-x', action='store_true',
                        default=False)

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as pfp:
            config = yaml.safe_load(pfp)

        parser.set_defaults(**config)

    # re-parse sys.argv to override defaults
    opts = parser.parse_args()

    if opts.dry:
        pprint.pprint(vars(opts))
        exit(0)

    if opts.model is None:
        print("Building model...")
        sim = Model(
            event_file_path=opts.events,
            infection_rate=opts.infection
        )
        print("Model built.")
        print("Initialising states...")
        sim.initialise_state()
        print("States intialised.")
        print("Generating movement events...")
        sim.generate_initial_events(permute=opts.permute_value)
        print("Movement events generated.")

        if opts.dump_model:
            print(f"Dumping initialised model to {opts.dump_file}...")
            with open(opts.dump_file, 'wb') as fp:
                pickle.dump(sim, fp)
            print(f"Model dumped.")
    else:
        print(f"Reading model from {opts.model}...")
        with open(opts.model, 'rb') as fp:
            sim = pickle.load(fp)
        print("Model reading finished.")
    print(f"N={len(sim.state.agents)} agents.")
    sim.infection_rate = opts.infection
    print("Choosing a seed...")
    sim.seed(fixed_seed=opts.seed)
    print("Infection seeding done.")
    print("Simulating...")
    sim.simulate(until=opts.maxtime, print_freq=opts.print_freq)   # 2196 is 6 years ish
    print("Simulation finished.")
    print("Recording results...")
    actual_outpath = sim.write_result(opts.output)
    opts.output = actual_outpath
    print(f"Recorded to {opts.output}.")
    print(f"Writing metadata to {opts.metadata}...")
    sim.write_metadata(opts.metadata, opts=opts)
    print("Metadata recorded.")
    print("Done.")
