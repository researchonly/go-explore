
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

import random
from .import_ai import *


class MontezumaPosLevel:
    __slots__ = ['level', 'score', 'room', 'x', 'y', 'tuple']

    def __init__(self, level, score, room, x, y):
        self.level = level
        self.score = score
        self.room = room
        self.x = x
        self.y = y

        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self.level, self.score, self.room, self.x, self.y)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, MontezumaPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self.level, self.score, self.room, self.x, self.y = d
        self.tuple = d

    def __repr__(self):
        return f'Level={self.level} Room={self.room} Objects={self.score} x={self.x} y={self.y}'

PYRAMID = [
    [-1, -1, -1, 0, 1, 2, -1, -1, -1],
    [-1, -1, 3, 4, 5, 6, 7, -1, -1],
    [-1, 8, 9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23]
]

OBJECT_PIXELS = [
    50,  # Hammer/mallet
    40,  # Key 1
    40,  # Key 2
    40,  # Key 3
    37,  # Sword 1
    37,  # Sword 2
    42   # Torch
]

KNOWN_XY = [None] * 24

KEY_BITS = 0x8 | 0x4 | 0x2


def get_room_xy(room):
    if KNOWN_XY[room] is None:
        for y, l in enumerate(PYRAMID):
            if room in l:
                KNOWN_XY[room] = (l.index(room), y)
                break
    return KNOWN_XY[room]


def clip(a, m, M):
    if a < m:
        return m
    if a > M:
        return M
    return a


class MyMontezuma:
    def __init__(self, check_death: bool = True, unprocessed_state: bool = False):  # TODO: version that also considers the room objects were found in
        # self.env = gym.make('MontezumaRevengeNoFrameskip-v4') # render_mode='human'
        self.env = gym.make('MontezumaRevengeDeterministic-v4') # render_mode='human'
        self.env.reset()
        self.ram = None
        self.check_death = check_death
        self.cur_steps = 0
        self.cur_score = 0
        self.rooms = {}
        self.room_time = None
        self.room_threshold = 40
        self.unwrapped.seed(0)
        self.unprocessed_state = unprocessed_state

    def __getattr__(self, e):
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        unprocessed_state = self.env.reset()
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = 0
        self.cur_steps = 0
        self.pos = None
        self.pos = self.pos_from_unprocessed_state(self.get_face_pixels(unprocessed_state), unprocessed_state)
        if self.get_pos().room not in self.rooms:
            self.rooms[self.get_pos().room] = (False, unprocessed_state[50:])
        self.room_time = (self.get_pos().room, 0)
        return unprocessed_state

    def pos_from_unprocessed_state(self, face_pixels, unprocessed_state):
        face_pixels = [(y, x) for y, x in face_pixels]
        if len(face_pixels) == 0:
            assert self.pos != None, 'No face pixel and no previous pos'
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        room = 1
        level = 0
        if self.pos is not None:
            room = self.pos.room
            level = self.pos.level
            direction_x = clip(int((self.pos.x - x) / 50), -1, 1)
            direction_y = clip(int((self.pos.y - y) / 50), -1, 1)
            if direction_x != 0 or direction_y != 0:
                # TODO(AE): shoudln't this call the static method?
                room_x, room_y = get_room_xy(self.pos.room)
                if room == 15 and room_y + direction_y >= len(PYRAMID):
                    room = 1
                    level += 1
                else:
                    assert direction_x == 0 or direction_y == 0, f'Room change in more than two directions : ({direction_y}, {direction_x})'
                    room = PYRAMID[room_y + direction_y][room_x + direction_x]
                    assert room != -1, f'Impossible room change: ({direction_y}, {direction_x})'

        # return MontezumaPosLevel(level, score, room, x, y)
        # plt.imshow(unprocessed_state)
        # plt.savefig(f'images/x={x}-ramx={self.ram[42]}-y={y}-ramy={260-self.ram[43]-random.random()}.png')
        # return MontezumaPosLevel(level, self.cur_score, room, x, y)
        return MontezumaPosLevel(level, self.cur_score, room, self.ram[42], 260 - self.ram[43])

    def get_restore(self):
        return (
            self.unwrapped.clone_state(),
            self.cur_score,
            self.cur_steps,
            self.pos,
            self.room_time,
        )

    def restore(self, data):
        (full_state, score, steps, pos, room_time) = data
        self.env._elapsed_steps = 0
        self.env._episode_started_at = time.time()

        self.unwrapped.restore_state(full_state)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = score
        self.cur_steps = steps
        self.pos = pos
        self.room_time = room_time

    def is_transition_screen(self, unprocessed_state):
        unprocessed_state = unprocessed_state[50:, :, :]
        # The screen is a transition screen if it is all black or if its color is made up only of black and
        # (0, 28, 136), which is a color seen in the transition screens between two levels.
        return (
                       np.sum(unprocessed_state[:, :, 0] == 0) +
                       np.sum((unprocessed_state[:, :, 1] == 0) | (unprocessed_state[:, :, 1] == 28)) +
                       np.sum((unprocessed_state[:, :, 2] == 0) | (unprocessed_state[:, :, 2] == 136))
               ) == unprocessed_state.size

    def get_face_pixels(self, unprocessed_state):
        return set(zip(*np.where(unprocessed_state[50:, :, 0] == 228)))

    def step(self, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        unprocessed_state, reward, done, lol = self.env.step(action)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_steps += 1

        face_pixels = self.get_face_pixels(unprocessed_state)
        pixel_death = self.ram[55] != 0

        if self.check_death and pixel_death:
            done = True

        self.cur_score += reward
        self.pos = self.pos_from_unprocessed_state(face_pixels, unprocessed_state)
        if self.pos.room != self.room_time[0]:
            self.room_time = (self.pos.room, 0)
        self.room_time = (self.pos.room, self.room_time[1] + 1)
        if (
            self.pos.room not in self.rooms or 
            (self.room_time[1] == self.room_threshold and not self.rooms[self.pos.room][0])
        ):
            self.rooms[self.pos.room] = (
                self.room_time[1] == self.room_threshold,
                unprocessed_state[50:]
            )
        return unprocessed_state, reward, done, lol

    def get_pos(self):
        assert self.pos is not None
        return self.pos

    @staticmethod
    def get_room_xy(room):
        if KNOWN_XY[room] is None:
            for y, l in enumerate(PYRAMID):
                if room in l:
                    KNOWN_XY[room] = (l.index(room), y)
                    break
        return KNOWN_XY[room]

    @staticmethod
    def get_room_out_of_bounds(room_x, room_y):
        return room_y < 0 or room_x < 0 or room_y >= len(PYRAMID) or room_x >= len(PYRAMID[0])

    @staticmethod
    def get_room_from_xy(room_x, room_y):
        return PYRAMID[room_y][room_x]

    @staticmethod
    def make_pos(score, pos):
        return MontezumaPosLevel(pos.level, score, pos.room, pos.x, pos.y)
