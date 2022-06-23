
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .explorers import *
from .montezuma_env import *
from .generic_atari_env import *
from .utils import *
import loky
import gzip
import bz2

compress = bz2
compress_suffix = '.bz2'
compress_kwargs = {}
n_digits = 20
DONE = None


class LPool:
    def __init__(self, n_cpus, maxtasksperchild=100):
        self.pool = loky.get_reusable_executor(n_cpus, timeout=100)

    def map(self, f, r):
        return self.pool.map(f, r)


class SyncPool:
    def __init__(self, n_cpus, maxtasksperchild=100):
        pass

    def map(self, f, r):
        res = []
        f_pickle = pickle.dumps(f)
        for e in r:
            e = pickle.loads(pickle.dumps(e))
            f2 = pickle.loads(f_pickle)
            res.append(f2(e))

        return res


def run_f_seeded(args):
    f, seed, args = args
    with use_seed(seed):
        return f(args)


class SeedPoolWrap:
    def __init__(self, pool):
        self.pool = pool

    def map(self, f, r):
        return self.pool.map(run_f_seeded, [(f, random.randint(0, 2**32 - 10), e) for e in r])


def seed_pool_wrapper(pool_class):
    def f(*args, **kwargs):
        return SeedPoolWrap(pool_class(*args, **kwargs))
    return f


class Discretizer:
    def __init__(self, attr, sort=False):
        self.attr = attr
        self.sort = sort
        self.cur_pos = None

    def apply(self, pos):
        self.cur_pos = pos
        obj = getattr(pos, self.attr)
        res = self.apply_rec(obj)
        self.cur_pos = None
        return res

    def apply_rec(self, obj):
        if isinstance(obj, (list, tuple, np.ndarray)):
            res = tuple(self.apply_rec(e) for e in obj)
            if self.sort:
                return tuple(sorted(res))
            else:
                return res
        return self.apply_scalar(obj)


class GridDimension(Discretizer):
    def __init__(self, attr, div, offset=0, sort=False):
        super().__init__(attr, sort=sort)
        self.div = div
        self.offset = offset

    def apply_scalar(self, scalar):
        if scalar is None or isinstance(scalar, str):
            return scalar
        if self.div == 1:
            return scalar
        return int(np.floor((scalar + self.offset) / self.div))

    def __repr__(self):
        return f'GridDimension("{self.attr}", {self.div}, {self.offset})'


class Cell:
    def __init__(self, score=-infinity, seen_times=0, chosen_times=0, 
                 trajectory_len=infinity, restore=None, exact_pos=None, real_cell=None):
        self.score = score

        self.seen_times = seen_times
        self.chosen_times = chosen_times

        self.trajectory_len = trajectory_len
        self.restore = restore
        self.exact_pos = exact_pos
        self.real_cell = real_cell

@dataclass
class PosInfo:
    __slots__ = ['exact', 'cell', 'state', 'restore']
    exact: tuple
    cell: tuple
    state: typing.Any
    restore: typing.Any


@dataclass
class TrajectoryElement:
    __slots__ = ['to', 'action', 'reward', 'done', 'real_pos']
    to: PosInfo
    action: int
    reward: float
    done: bool
    real_pos: MontezumaPosLevel


Experience = tuple

POOL = None
ENV = None

def get_env():
    return ENV


def get_downscale(args):
    f, cur_shape, cur_pix_val = args
    return imdownscale(f, cur_shape, cur_pix_val).flatten().tobytes()


@functools.lru_cache(maxsize=1)
def get_saved_grid(file):
    return pickle.load(compress.open(file, 'rb'))


class Explore:
    def __init__(
            self, explorer_policy, cell_selector, env,
            grid_info: tuple,
            pool_class,
            args,
            important_attrs
    ):
        global POOL, ENV
        self.args = args
        self.important_attrs = important_attrs

        self.prev_checkpoint = None
        self.env_info = env
        self.make_env()
        self.pool_class = pool_class
        if self.args.reset_pool:
            # POOL = self.pool_class(multiprocessing.cpu_count() // 2)
            POOL = self.pool_class(1)
        else:
            # POOL = self.pool_class(multiprocessing.cpu_count() // 2, maxtasksperchild=100)
            POOL = self.pool_class(1, maxtasksperchild=1)

        self.explorer = explorer_policy
        self.selector = cell_selector
        self.grid_info = grid_info
        self.grid = defaultdict(Cell)
        self.frames_true = 0
        self.frames_compute = 0
        self.cycles = 0
        self.max_score = 0

        self.state = None
        self.reset()

        self.normal_frame_shape = (160, 210)
        cell_key = self.get_cell()
        self.grid[cell_key] = Cell()
        self.grid[cell_key].trajectory_len = 0
        self.grid[cell_key].score = 0
        self.grid[cell_key].exact_pos = self.get_pos()
        self.grid[cell_key].real_cell = self.get_real_cell()
        # Create the DONE cell
        self.grid[DONE] = Cell()
        self.selector.cell_update(cell_key, self.grid[cell_key])
        self.selector.cell_update(DONE, self.grid[DONE])
        self.real_cell = None

    def make_env(self):
        global ENV
        if ENV is None:
            ENV = self.env_info[0](**self.env_info[1])
            ENV.reset()

    def reset(self):
        self.real_cell = None
        self.make_env()
        return ENV.reset()

    def step(self, action):
        self.real_cell = None
        return ENV.step(action)

    def get_pos(self):
        return self.get_real_pos()

    def get_real_pos(self):
        return ENV.get_pos()

    def get_pos_info(self, include_restore=True):
        return PosInfo(
            self.get_pos() if self.args.use_real_pos else None,
            self.get_cell(),
            None,
            self.get_restore() if include_restore else None,
        )

    def get_restore(self):
        return ENV.get_restore()

    def restore(self, val):
        self.real_cell = None
        self.make_env()
        ENV.restore(val)

    def get_real_cell(self):
        if self.real_cell is None:
            pos = self.get_real_pos()
            res = {}
            for dimension in self.grid_info:
                res[dimension.attr] = dimension.apply(pos)
            self.real_cell = pos.__class__(**res)
        return self.real_cell

    def get_cell(self):
        if self.args.use_real_pos:
            return self.get_real_cell()
        else:
            pos = self.get_pos()
            return pos

    def run_explorer(self, explorer, start_cell=None, max_steps=-1):
        trajectory = []
        while True:
            if ((max_steps > 0 and len(trajectory) >= max_steps)):
                break
            action = explorer.get_action(ENV)
            state, reward, done, _ = self.step(action)
            self.frames_true += 1
            self.frames_compute += 1
            trajectory.append(
                TrajectoryElement(
                    # initial_pos_info,
                    self.get_pos_info(),
                    action, reward, done,
                    self.get_real_cell(),
                )
            )
            if done:
                break
        return trajectory

    def run_seed(self, seed, start_cell=None, max_steps=-1):
        with use_seed(seed):
            self.explorer.init_seed()
            return self.run_explorer(self.explorer, start_cell, max_steps)

    def process_cell(self, info):
        # This function runs in a SUBPROCESS, and processes a single cell.
        cell_key, cell, seed, known_rooms, target_shape, max_pix = info.data
        assert cell_key != DONE
        self.env_info[0].TARGET_SHAPE = target_shape
        self.env_info[0].MAX_PIX_VALUE = max_pix
        self.frames_true = 0
        self.frames_compute = 0

        if cell.restore is not None:
            self.restore(cell.restore)
            self.frames_true += cell.trajectory_len
        else:
            assert cell.trajectory_len == 0, 'Cells must have a restore unless they are the initial state'
            self.reset()

        end_trajectory = self.run_seed(seed, max_steps=self.args.explore_steps)

        known_room_data = {}
        if len(ENV.rooms) > known_rooms:
            known_room_data = ENV.rooms

        return TimedPickle((cell_key, end_trajectory, self.frames_true, self.frames_compute, known_room_data), 'ret', enabled=info.enabled)

    def run_cycle(self):
        # Choose a bunch of cells, send them to the workers for processing, then combine the results.
        # A lot of what this function does is only aimed at minimizing the amount of data that needs
        # to be pickled to the workers, which is why it sets a lot of variables to None only to restore
        # them later.
        global POOL
        
        self.cycles += 1
        chosen_cells = []
        cell_keys = self.selector.choose_cell(self.grid, size=self.args.batch_size)
        for i, cell_key in enumerate(cell_keys):
            cell_copy = self.grid[cell_key]
            seed = random.randint(0, 2 ** 31)
            chosen_cells.append(TimedPickle((cell_key, cell_copy, seed,
                                             len(ENV.rooms), self.env_info[0].TARGET_SHAPE,
                                             self.env_info[0].MAX_PIX_VALUE), 'args', enabled=(i == 0 and False)))

        # NB: save some of the attrs that won't be necessary but are very large, and set them to none instead,
        #     this way they won't be pickled.
        cache = {}
        to_save = [
            'grid', 'selector', 'pool_class'
        ]
        for attr in to_save:
            cache[attr] = getattr(self, attr)
            setattr(self, attr, None)

        trajectories = [e.data for e in POOL.map(self.process_cell, chosen_cells)]
        if self.args.reset_pool and (self.cycles + 1) % 100 == 0:
            POOL.close()
            POOL.join()
            POOL = None
            gc.collect()
            POOL = self.pool_class(self.args.n_cpus)
        chosen_cells = [e.data for e in chosen_cells]

        for attr, v in cache.items():
            setattr(self, attr, v)
        
        # Note: we do this now because starting here we're going to be concatenating the trajectories
        # of these cells, and they need to remain the same!
        chosen_cells = [(k, copy.copy(c), s, n, shape, pix) for k, c, s, n, shape, pix in chosen_cells]
        cells_to_reset = set()

        for ((cell_key, cell_copy, seed, _, _, _), (_, end_trajectory, ft, fc, known_rooms)) in zip(chosen_cells,
                                                                                                  trajectories):
            self.frames_true += ft
            self.frames_compute += fc
            seen_cells = set([cell_key])

            for k in known_rooms:
                if k not in ENV.rooms:
                    ENV.rooms[k] = known_rooms[k]

            start_cell = self.grid[cell_key]
            start_cell.chosen_times += 1
            start_cell.seen_times += 1
            self.selector.cell_update(cell_key, start_cell)
            cur_score = cell_copy.score
            potential_cell = start_cell
            old_potential_cell_key = cell_key
            for i, elem in enumerate(end_trajectory):
                potential_cell_key = elem.to.cell
                if elem.done:
                    potential_cell_key = DONE

                if potential_cell_key != old_potential_cell_key:
                    potential_cell = self.grid[potential_cell_key]
                    if potential_cell_key not in seen_cells:
                        seen_cells.add(potential_cell_key)
                        potential_cell.seen_times += 1
                        self.selector.cell_update(potential_cell_key, potential_cell)

                full_traj_len = cell_copy.trajectory_len + i + 1
                cur_score += elem.reward

                # Note: the DONE element should have a 0% chance of being selected, so OK to add the cell if it is in the DONE state.
                if (elem.to.restore is not None or potential_cell_key == DONE) and self.should_accept_cell(potential_cell, cur_score, full_traj_len):
                    cells_to_reset.add(potential_cell_key)
                    potential_cell.trajectory_len = full_traj_len
                    potential_cell.restore = elem.to.restore
                    assert potential_cell.restore is not None or potential_cell_key == DONE
                    potential_cell.score = cur_score
                    if cur_score > self.max_score:
                        self.max_score = cur_score
                    potential_cell.real_cell = elem.real_pos
                    if self.args.use_real_pos:
                        potential_cell.exact_pos = elem.to.exact

                    self.selector.cell_update(potential_cell_key, potential_cell)

        if self.args.reset_cell_on_update:
            for cell_key in cells_to_reset:
                self.grid[cell_key].chosen_times = 0
                self.grid[cell_key].seen_times = 0

        return [(k) for k, c, s, n, shape, pix in chosen_cells], trajectories

    def should_accept_cell(self, potential_cell, cur_score, full_traj_len):
        if self.args.prob_override > 0.0000000001 and random.random() < self.args.prob_override:
            return True
        if self.args.optimize_score:
            return (cur_score > potential_cell.score or
                    (full_traj_len < potential_cell.trajectory_len and
                     cur_score == potential_cell.score))
        return full_traj_len < potential_cell.trajectory_len
