#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



results=${2:-results}
frames=${3:-250000000}

python goexplore_py/main.py --chosen_weight=1 --remember_rooms --reset_cell_on_update --game=montezuma --max_hours=256 --max_compute_steps=$frames --base_path=$results

