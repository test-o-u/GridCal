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
    # 'delta': 1.888852339,
    # 'omega': 2

}

Y0 = {
    'a1': 15 * (np.pi / 180),  # rotor angle (rad)
    'a2': 10 * (np.pi / 180),  # angle of second bus
    'v1': 1.0,                 # generator terminal voltage magnitude (pu)
    'v2': 0.95,                # remote bus voltage (pu)
    'psid': 1.0,  # d-axis flux linkage (pu)
    'psiq': 0.0,  # q-axis flux linkage (pu)
    'i_d': 0.1,  # d-axis stator current (pu)
    'i_q': 0.2,  # q-axis stator current (pu)
    'vd': 0.0,  # d-axis voltage (pu)
    'vq': 1.0,  # q-axis voltage (pu)
    'te': 0.1,  # electromagnetic torque (pu)
    'Pe': 0.2,  # real power (pu)
    'Qe': 0.2,  # reactive power (pu)
    # 'a1': 1,
    # 'a2': 0.9149,
    # 'v1': 2,
    # 'v2': 1.95877996,
    # 'psid': 1.63030343,
    # 'psiq':0.77634888,
    # 'i_d': 4.16194885,
    # 'i_q': 1.91335163,
    # 'vd': 1.77634888,
    # 'vq': 1.63030343,
    # 'te': 4.03046412,
    # 'Pe': 4.03046412,
    # 'Qe': 2.28390768

}

DAE_X0 = np.array(list(X0.values()))
DAE_Y0 = np.array(list(Y0.values()))

# Simulation parameters
SIMULATION_TIME = 10 # Total simulation time (seconds)
TIME_STEP = 0.01        # Time step for simulation (seconds)
TOL = 1e-5             # Tolerance for numerical methods
MAX_ITER = 1000           # Maximum iterations for numerical methods
# Simulation methods
STEADYSTATE_METHOD="steadystate" 
INTEGRATION_METHOD="trapezoid"

# Performance boolean
PERFORMANCE = False  # Set to True for performance testing, False for debugging