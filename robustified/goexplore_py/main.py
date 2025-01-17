
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from sys import platform
from goexplore_py.randselectors import *
from goexplore_py.goexplore import *

VERSION = 1

THRESH_TRUE = 20_000_000_000
THRESH_COMPUTE = 1_000_000
MAX_FRAMES = None
MAX_FRAMES_COMPUTE = None
MAX_ITERATIONS = None
MAX_TIME = 12 * 60 * 60
MAX_CELLS = None
MAX_SCORE = None

def _run(args):

    explorer = RandomExplorer()
    # explorer = RepeatedRandomExplorer(mean_repeat=20)

    if args.use_real_pos:
        args.target_shape = None
        args.max_pix_value = None

    if args.dynamic_state:
        args.target_shape = (-1, -1)
        args.max_pix_value = -1

    if args.game == 'montezuma':
        game_class = MyMontezuma
        game_class.TARGET_SHAPE = args.target_shape
        game_class.MAX_PIX_VALUE = args.max_pix_value
        game_args = dict()
        grid_resolution = (
            GridDimension('level', 1), GridDimension('score', 1), GridDimension('room', 1),
            GridDimension('x', args.resolution), GridDimension('y', args.resolution)
        )
    else:
        raise NotImplementedError("Unknown game: " + args.game)

    selector = WeightedSelector(game_class)

    pool_cls = multiprocessing.get_context(args.start_method).Pool
    pool_cls = LPool

    pool_cls = seed_pool_wrapper(pool_cls)

    expl = Explore(
        explorer,
        selector,
        (game_class, game_args),
        grid_resolution,
        pool_class=pool_cls,
        args=args
    )

    with tqdm(desc='Time (seconds)', smoothing=0, total=MAX_TIME) as t_time, \
            tqdm(desc='Iterations', total=MAX_ITERATIONS) as t_iter, \
            tqdm(desc='Compute steps', total=MAX_FRAMES_COMPUTE) as t_compute, \
            tqdm(desc='Game step', total=MAX_FRAMES) as t, \
            tqdm(desc='Max score', total=MAX_SCORE) as t_score, \
            tqdm(desc='Done score', total=MAX_SCORE) as t_done_score, \
            tqdm(desc='Cells', total=MAX_CELLS) as t_cells:
        t_compute.update(expl.frames_compute)
        t.update(expl.frames_true)
        start_time = time.time()
        last_time = np.round(start_time)
        n_iters = 0

        def should_continue():
            if MAX_TIME is not None and time.time() - start_time >= MAX_TIME:
                return False
            if MAX_FRAMES is not None and expl.frames_true >= MAX_FRAMES:
                return False
            if MAX_FRAMES_COMPUTE is not None and expl.frames_compute >= MAX_FRAMES_COMPUTE:
                return False
            if MAX_ITERATIONS is not None and n_iters >= MAX_ITERATIONS:
                return False
            if MAX_CELLS is not None and len(expl.grid) >= MAX_CELLS:
                return False
            if MAX_SCORE is not None and expl.max_score >= MAX_SCORE:
                return False
            return True

        while should_continue():
            # Run one iteration
            old = expl.frames_true
            old_compute = expl.frames_compute
            old_max_score = expl.max_score

            expl.run_cycle()

            t.update(expl.frames_true - old)
            t_score.update(expl.max_score - old_max_score)
            t_done_score.n = expl.grid[DONE].score
            t_done_score.refresh()
            t_compute.update(expl.frames_compute - old_compute)
            t_iter.update(1)
            # Note: due to the archive compression that can happen with dynamic cell representation,
            # we need to do this so that tqdm doesn't complain about negative updates.
            t_cells.n = len(expl.grid)
            t_cells.refresh()

            cur_time = np.round(time.time())
            t_time.update(int(cur_time - last_time))
            last_time = cur_time
            n_iters += 1

class Tee(object):
    def __init__(self, name, output):
        self.file = open(name, 'w')
        self.stdout = getattr(sys, output)
        self.output = output
        setattr(sys, self.output, self)
    def __del__(self):
        setattr(sys, self.output, self.stdout)
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
    
def run(base_path, args):
    cur_id = 0

    base_path = f'{base_path}/{cur_id:04d}_{uuid.uuid4().hex}/'
    args.base_path = base_path
    os.makedirs(base_path, exist_ok=True)
    open(f'{base_path}/thisisfake_{args.max_compute_steps}_experience.bz2', 'w')
    info = copy.copy(vars(args))
    info['version'] = VERSION
    info['code_hash'] = get_code_hash()
    print('Code hash:', info['code_hash'])
    del info['base_path']
    json.dump(info, open(base_path + '/kwargs.json', 'w'), sort_keys=True, indent=2)

    code_path = base_path + '/code'
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    shutil.copytree(cur_dir, code_path, ignore=shutil.ignore_patterns('*.png', '*.stl', '*.JPG', '__pycache__', 'LICENSE*', 'README*'))

    teeout = Tee(args.base_path + '/log.out', 'stdout')
    teeerr = Tee(args.base_path + '/log.err', 'stderr')
    
    print('Experiment running in', base_path)

    try:
        _run(args)
    except Exception as e:
        import traceback
        print(e)
        traceback.print_exc()
        import signal
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            os.kill(child.pid, signal.SIGTERM)
        open(base_path + 'has_died', 'w')
        os._exit(1)

if __name__ == '__main__':

    if platform == "darwin":
        # Circumvents the following issue on Mac OS:
        # https://github.com/opencv/opencv/issues/5150
        cv2.setNumThreads(0)
    
    parser = argparse.ArgumentParser()

    current_group = parser

    def boolarg(arg, *args, default=False, help='', neg=None, dest=None):
        def extract_name(a):
            dashes = ''
            while a[0] == '-':
                dashes += '-'
                a = a[1:]
            return dashes, a

        if dest is None:
            _, dest = extract_name(arg)

        group = current_group.add_mutually_exclusive_group()
        group.add_argument(arg, *args, dest=dest, action='store_true', help=help + (' (DEFAULT)' if default else ''), default=default)
        not_args = []
        for a in [arg] + list(args):
            dashes, name = extract_name(a)
            not_args.append(f'{dashes}no_{name}')
        if isinstance(neg, str):
            not_args[0] = neg
        if isinstance(neg, list):
            not_args = neg
        group.add_argument(*not_args, dest=dest, action='store_false', help=f'Opposite of {arg}' + (' (DEFAULT)' if not default else ''), default=default)

    def add_argument(*args, **kwargs):
        if 'help' in kwargs and kwargs.get('default') is not None:
            kwargs['help'] += f' (default: {kwargs.get("default")})'

        current_group.add_argument(*args, **kwargs)

    current_group = parser.add_argument_group('General Go-Explore')
    add_argument('--game', '-g', type=str, default='montezuma', help='Determines the game to which apply goexplore.')
    add_argument('--repeat_action', '--ra', type=float, default=20, help='The average number of times that actions will be repeated in the exploration phase.')
    add_argument('--explore_steps', type=int, default=100, help='Maximum number of steps in the explore phase.')
    boolarg('--optimize_score', default=True, help='Optimize for score (only speed). Will use fewer "game frames" and come up with faster trajectories with lower scores. If not combined with --remember_rooms and --objects_from_ram is not enabled, things should run much slower.')
    add_argument('--prob_override', type=float, default=0.0, help='Probability that the newly found cells will randomly replace the current cell.')
    add_argument('--batch_size', type=int, default=100, help='Number of worker threads to spawn')
    boolarg('--reset_cell_on_update', '--rcou', help='Reset the times-chosen and times-chosen-since when a cell is updated.')
    add_argument('--explorer_type', type=str, default='repeated', help='The type of explorer. repeated, drift or random.')
    add_argument('--seed', type=int, default=None, help='The random seed.')

    current_group = parser.add_argument_group('Checkpointing')
    add_argument('--base_path', '-p', type=str, default='./results/', help='Folder in which to store results')
    add_argument('--path_postfix', '--pf', type=str, default='', help='String appended to the base path.')
    add_argument('--checkpoint_game', type=int, default=20_000_000_000_000_000_000, help='Save a checkpoint every this many GAME frames (note: recommmended to ignore, since this grows very fast at the end).')
    add_argument('--checkpoint_compute', type=int, default=1_000_000, help='Save a checkpoint every this many COMPUTE frames.')
    boolarg('--pictures', dest='save_pictures', help='Save pictures of the pyramid every checkpoint (uses more space).')
    boolarg('--prob_pictures', '--pp', dest='save_prob_pictures', help='Save pictures of showing probabilities.')
    boolarg('--item_pictures', '--ip', dest='save_item_pictures', help='Save pictures of showing items collected.')
    boolarg('--clear_old_checkpoints', neg='--keep_checkpoints', default=True,
            help='Clear large format checkpoints. Checkpoints aren\'t necessary for view folder to work. They use a lot of space.')
    boolarg('--keep_prob_pictures', '--kpp', help='Keep old pictures showing probabilities.')
    boolarg('--keep_item_pictures', '--kip', help='Keep old pictures showing items collected.')
    boolarg('--warn_delete', default=True, help='Warn before deleting the existing directory, if any.')
    boolarg('--save_cells', default=False, help='Save exact cells produced by Go-Explore instead of just hints as to whether they are done or not.')

    current_group = parser.add_argument_group('Runtime')
    add_argument('--max_game_steps', type=int, default=None, help='Maximum number of GAME frames.')
    add_argument('--max_compute_steps', '--mcs', type=int, default=None, help='Maximum number of COMPUTE frames.')
    add_argument('--max_iterations', type=int, default=None, help='Maximum number of iterations.')
    add_argument('--max_hours', '--mh', type=float, default=12, help='Maximum number of hours to run this for.')
    add_argument('--max_cells', type=int, default=None, help='The maximum number of cells before stopping.')
    add_argument('--max_score', type=float, default=None, help='Stop when this score (or more) has been reached in the archive.')

    current_group = parser.add_argument_group('General Selection Probability')
    add_argument('--seen_weight', '--sw', type=float, default=0.0, help='The weight of the "seen" attribute in cell selection.')
    add_argument('--seen_power', '--sp', type=float, default=0.5, help='The power of the "seen" attribute in cell selection.')
    add_argument('--chosen_weight', '--cw', type=float, default=0.0, help='The weight of the "chosen" attribute in cell selection.')
    add_argument('--chosen_power', '--cp', type=float, default=0.5, help='The power of the "chosen" attribute in cell selection.')
    add_argument('--chosen_since_new_weight', '--csnw', type=float, default=0.0, help='The weight of the "chosen since new" attribute in cell selection.')
    add_argument('--chosen_since_new_power', '--csnp', type=float, default=0.5, help='The power of the "chosen since new" attribute in cell selection.')
    add_argument('--action_weight', '--aw', type=float, default=0.0, help='The weight of the "action" attribute in cell selection.')
    add_argument('--action_power', '--ap', type=float, default=0.5, help='The power of the "action" attribute in cell selection.')

    current_group = parser.add_argument_group('Atari Domain Knowledge')
    add_argument('--resolution', '--res', type=float, default=16, help='Length of the side of a grid cell.')
    boolarg('--use_objects', neg='--use_scores', default=True, help='Use objects in the cell description. Otherwise scores will be used.')
    add_argument('--pitfall_treasure_type', type=str, default='none', help='How to include treasures in the cell description of Pitfall: none (don\'t include treasures), count (include treasure count), score (include sum of positive rewards) or location (include the specific location the treasures were found).')

    current_group = parser.add_argument_group('Atari No Domain Knowledge')
    boolarg('--use_real_pos', neg=['--state_is_pixels', '--pix'], default=True, help='If this is on, the state will be resized pixels, not human prior.')
    add_argument('--resize_x', '--rx', type=int, default=11, help='What to resize the pixels to in the x direction for use as a state.')
    add_argument('--resize_y', '--ry', type=int, default=8, help='What to resize the pixels to in the y direction for use as a state.')
    add_argument('--max_pix_value', '--mpv', type=int, default=8, help='The range of pixel values when resizing will be rescaled to from 0 to this value. Lower means fewer possible states in states_is_pixels.')
    add_argument('--resize_shape', type=str, default=None, help='Shortcut for passing --resize_x (0), --resize_y (1) and --max_pix_value (2) all at the same time: 0x1x2')

    boolarg('--dynamic_state', help='Dynamic downscaling of states. Ignores --resize_x, --resize_y, --max_pix_value and --resize_shape.')

    add_argument('--first_compute_dynamic_state', type=int, default=100_000, help='Number of steps before recomputing the dynamic state representation (ignored if negative).')
    add_argument('--first_compute_archive_size', type=int, default=10_000, help='Number of steps before recomputing the dynamic state representation (ignored if negative).')
    add_argument('--recompute_dynamic_state_every', type=int, default=5_000_000, help='Number of steps before recomputing the dynamic state representation (ignored if negative).')
    add_argument('--max_archive_size', type=int, default=1_000_000_000, help='Number of steps before recomputing the dynamic state representation (ignored if negative).')

    add_argument('--cell_split_factor', type=float, default=0.03, help='The factor by which we try to split frames when recomputing the representation. 1 -> each frame is its own cell. 0 -> all frames are in the same cell.')
    add_argument('--split_iterations', type=int, default=100, help='The number of iterations when recomputing the representation. A higher number means a more accurate (but less stochastic) results, and a lower number means a more stochastic and less accurate result. Note that stochasticity can be a good thing here as it makes it harder to get stuck.')
    add_argument('--max_recent_frames', type=int, default=5_000, help='The number of recent frames to use in recomputing the representation. A higher number means slower recomputation but more accuracy, a lower number is faster and more stochastic.')
    add_argument('--recent_frame_add_prob', type=float, default=0.1, help='The probability for a frame to be added to the list of recent frames.')

    current_group = parser.add_argument_group('OpenAI Robotics')
    add_argument('--interval_size', type=float, default=0.1, help='The interval size for robotics envs.')

    current_group = parser.add_argument_group('Fetch Robotics')
    add_argument('--fetch_type', type=str, default='boxes', help='The type of fetch environment (boxes, cubes, objects...)')
    add_argument('--nsubsteps', type=int, default=20, help='The number of substeps in mujoco between each action (each substep takes 0.002 seconds).')
    add_argument('--target_location', type=str, default=None, help='The target location for fetch envs.')
    add_argument('--min_grip_score', type=int, default=0, help='The minimum grip score (inclusive) for a fetch grip to be included in the archive.\n0: at least 1 finger touching, 1: 2 fingers touching, 3: 2 fingers touching AND not touching the table (gripping and lifting).')
    add_argument('--max_grip_score', type=int, default=3, help='The maximum grip score (inclusive). All grips with higher scores will be given this score instead.')
    add_argument('--minmax_grip_score', type=str, default=None, help='Shortcut to set both the min and max grip score. The first digit is the min and second is the max.')
    add_argument('--door_resolution', type=float, default=0.2, help='Number by which to divide the door distance.')
    old_group = current_group
    # current_group = current_group.add_mutually_exclusive_group()
    add_argument('--timestep', type=float, default=0.002, help='The size of a mujoco timestep.')
    add_argument('--total_timestep', type=float, default=None, help='The total timestep length (if included, timestep is ignored from the command line and instead set to total_timestep / nsubsteps). A reasonable value is 0.08')
    current_group = old_group
    add_argument('--gripper_pos_resolution', type=float, default=0.5, help='Number by which to divide the gripper position.')
    add_argument('--door_weight', type=float, default=1.0, help='Weight of different door positions.')
    add_argument('--grip_weight', type=float, default=1.0, help='Weight of different grip positions.')
    boolarg('--fetch_uniform', help='Select uniformly for fetch. Shurtcut for --door_weight=0 --grip_weight=0 --low_level_weight=1.')
    boolarg('--conflate_objects', help='Conflate objects when getting their positions. With this, there is no difference between object 1 being in shelf 0001 and object 2 being in shelf 0001.')
    boolarg('--target_single_shelf', help='As soon as a shelf is reached, only target that one shelf going forward.')
    boolarg('--fetch_ordered', help='Whether to put objects in the shelves in a specific order or not.')
    boolarg('--combine_table_shelf_box', help='Combine the table and shelf box for determining death by being outside the table-shelf box.')
    boolarg('--fetch_force_closed_doors', help='Only give rewards for fetch if the doors are closed.')
    boolarg('--fetch_single_cell', help='Only one cell when doing fetch.', default=False)

    current_group = parser.add_argument_group('Performance')
    add_argument('--n_cpus', type=int, default=None, help='Number of worker threads to spawn')
    add_argument('--pool_class', type=str, default='loky', help='The multiprocessing pool class (py or torch or loky).')
    add_argument('--start_method', type=str, default='fork', help='The process start method.')
    boolarg('--reset_pool', help='The pool should be reset every 100 iterations.')

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed + 1)

    if args.total_timestep is not None:
        args.timestep = args.total_timestep / args.nsubsteps
    del args.total_timestep

    if args.fetch_uniform:
        args.door_weight = 0
        args.grip_weight = 0
        args.low_level_weight = 1
    del args.fetch_uniform

    if args.minmax_grip_score:
        args.min_grip_score = int(args.minmax_grip_score[0])
        args.max_grip_score = int(args.minmax_grip_score[1])
    del args.minmax_grip_score

    if args.resize_shape:
        x, y, p = args.resize_shape.split('x')
        args.resize_x = int(x)
        args.resize_y = int(y)
        args.max_pix_value = int(p)

    args.target_shape = (args.resize_x, args.resize_y)
    del args.resize_shape
    del args.resize_x
    del args.resize_y

    if args.start_method == 'fork' and args.pool_class == 'torch':
        raise Exception('Fork start method not supported by torch.multiprocessing.')

    THRESH_TRUE = args.checkpoint_game
    THRESH_COMPUTE = args.checkpoint_compute
    MAX_FRAMES = args.max_game_steps
    MAX_FRAMES_COMPUTE = args.max_compute_steps
    MAX_TIME = args.max_hours * 3600
    MAX_ITERATIONS = args.max_iterations
    MAX_CELLS = args.max_cells
    MAX_SCORE = args.max_score

    run(args.base_path, args)
