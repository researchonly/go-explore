
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .import_ai import *
from goexplore_py.goexplore import DONE


@dataclass()
class Weight:
    weight: float = 1.0
    power: float = 1.0

def numberOfSetBits(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24

def convert_score(e):
    # TODO: this doesn't work when actual score is used!! Fix?
    if isinstance(e, tuple):
        return len(e)
    return numberOfSetBits(e)


class WeightedSelector:
    def __init__(self, game):
        self.chosen = Weight(weight=1, power=0.5)
        self.game = game

        self.clear_all_cache()

    def clear_all_cache(self):
        self.all_weights = []
        self.cells = []
        self.cell_pos = {}
        self.cell_score = {}
        self.cached_pos_weights = {}
        self.possible_scores = defaultdict(int)
        self.known_object_pos = set()

    def get_score(self, cell_key, cell):
        if cell_key == DONE:
            return 0.0
        elif not isinstance(cell_key, tuple):
            return cell_key.score
        else:
            return cell.score

    def cell_update(self, cell_key, cell):
        if cell_key not in self.cell_pos:
            self.cell_pos[cell_key] = len(self.all_weights)
            self.all_weights.append(0.0)
            self.cells.append(cell_key)
            if cell_key != DONE:
                self.cell_score[cell_key] = self.get_score(cell_key, cell)
                self.possible_scores[self.get_score(cell_key, cell)] += 1
        elif cell_key != DONE:
            score = self.get_score(cell_key, cell)
            old_score = self.cell_score[cell_key]
            self.possible_scores[score] += 1
            self.possible_scores[old_score] -= 1
            self.cell_score[cell_key] = score
            if self.possible_scores[old_score] == 0:
                del self.possible_scores[old_score]

    def compute_weight(self, value, weight):
        return weight.weight * 1 / (value + 1) ** weight.power

    def get_chosen_weight(self, cell):
        return self.compute_weight(cell.chosen_times, self.chosen)

    def get_weight(self, cell_key, cell):
        if cell_key == DONE:
            return 0.0
        res = self.get_chosen_weight(cell)
        return res

    def update_weights(self, known_cells):
        for cell in self.cells:
            idx = self.cell_pos[cell]
            self.all_weights[idx] = self.get_weight(cell, known_cells[cell])

    def choose_cell(self, known_cells, size=1):
        self.update_weights(known_cells)
        weights = np.array(self.all_weights)
        total = np.sum(weights)
        idxs = np.random.choice(
            list(range(len(self.cells))),
            size=size,
            p=weights / total
        )
        # TODO: in extremely rare cases, we do select the DONE cell. Not sure why. We filter it out here but should
        # try to fix the underlying bug eventually.
        return [self.cells[i] for i in idxs if self.cells[i] != DONE]
