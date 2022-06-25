
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .import_ai import *
from goexplore_py.goexplore import DONE

class WeightedSelector:
    def __init__(self, game):
        self.game = game
        self.clear_all_cache()

    def clear_all_cache(self):
        self.all_weights = []
        self.cells = []
        self.cell_pos = {}

    def cell_update(self, cell_key, cell):
        if cell_key not in self.cell_pos:
            self.cell_pos[cell_key] = len(self.all_weights)
            self.all_weights.append(0.0)
            self.cells.append(cell_key)

    def get_weight(self, cell_key, cell):
        if cell_key == DONE:
            return 0.0
        res = 1 / (cell.chosen_times + 1) ** 0.5
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
