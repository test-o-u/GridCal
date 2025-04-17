# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

# Path to the system JSON
SYSTEM_JSON_PATH = "GridCalEngine/Devices/Dynamic/io/data/test.json"

# Models present in the Dynamic Simulation Engine
MODELS = list([
    ('bus', ['Bus']),
    ('line', ['ACLine']), 
    ('load', ['ExpLoad']),
    ('syngen', ['GENCLS'])
])

# Initial DAE state vectors
X0 = {
    'delta': 0.0,
    'omega': 1.0,
}

Y0 = {
    'a1': 15 * (np.pi / 180),  # rotor angle (rad)
    'a2': 10 * (np.pi / 180),  # angle of second bus
    'v1': 1.0,                 # generator terminal voltage magnitude (pu)
    'v2': 0.95,                # remote bus voltage (pu)
    'psid': 1.0,               # d-axis flux linkage (pu)
    'psiq': 0.0,               # q-axis flux linkage (pu)
    'i_d': 0.1,                # d-axis stator current (pu)
    'i_q': 0.2,                # q-axis stator current (pu)
    'vd': 0.0,                 # d-axis voltage (pu)
    'vq': 1.0,                 # q-axis voltage (pu)
    'te': 0.2,                 # electromagnetic torque (pu)
    'Pe': 0.2,                 # real power (pu)
    'Qe': 0.2,                 # reactive power (pu)
}

DAE_X0 = np.array(list(X0.values()))
DAE_Y0 = np.array(list(Y0.values()))

# Simulation parameters
SIMULATION_TIME = 10  # Total simulation time (seconds)
TIME_STEP = 0.01        # Time step for simulation (seconds)
TOL = 1e-5              # Tolerance for numerical methods
MAX_ITER = 80           # Maximum iterations for numerical methods
# Simulation methods
STEADYSTATE_METHOD="steadystate" 
INTEGRATION_METHOD="trapezoid"

# Performance boolean
PERFORMANCE = False  # Set to True for performance testing, False for debugging