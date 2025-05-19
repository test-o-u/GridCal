# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

# Path to the system and event JSON
SYSTEM_JSON_PATH = "GridCalEngine/Devices/Dynamic/io/data/test.json"
EVENTS_JSON_PATH = "GridCalEngine/Devices/Dynamic/io/event/events.json"

# Models present in the Dynamic Simulation Engine
MODELS = list([
    ('bus', ['Bus']),
    ('line', ['ACLine']), 
    ('load', ['ExpLoad']),
    ('syngen', ['GENCLS'])
])

# Initial DAE state vectors
X0 = {
    # 'delta': 0.0,
    # 'omega': 1.0,
    'delta': 0.00579736,
    'omega': 1.0

}

Y0 = {
    # 'a1': 15 * (np.pi / 180),  # rotor angle (rad)
    # 'a2': 10 * (np.pi / 180),  # angle of second bus
    # 'v1': 1.0,                 # generator terminal voltage magnitude (pu)
    # 'v2': 0.95,                # remote bus voltage (pu)
    # 'psid': 1.0,  # d-axis flux linkage (pu)
    # 'psiq': 0.0,  # q-axis flux linkage (pu)
    # 'i_d': 0.1,  # d-axis stator current (pu)
    # 'i_q': 0.2,  # q-axis stator current (pu)
    # 'vd': 0.0,  # d-axis voltage (pu)
    # 'vq': 1.0,  # q-axis voltage (pu)
    # 'te': 0.1,  # electromagnetic torque (pu)
    # 'Pe': 0.2,  # real power (pu)
    # 'Qe': 0.2,  # reactive power (pu)
    'a1': 2.06469935e-18,
    'a2': 1.95628959e-02,
    'v1': 1.0,
    'v2': 0.76775572,
    'psid': 1.00032224,
    'psiq': -0.03390404,
    'i_d': 18.73780599,
    'i_q': 0.22602691,
    'v_d': 0.00579733,
    'v_q': 0.9999832,
    't_e': 0.86138701,
    'P_e': 0.33465232,
    'Q_e': 18.73618076

}

DAE_X0 = np.array(list(X0.values()))
DAE_Y0 = np.array(list(Y0.values()))

# Simulation parameters
SIMULATION_TIME = 3 # Total simulation time (seconds)
TIME_STEP = 0.01        # Time step for simulation (seconds)
TOL = 1e-10            # Tolerance for numerical methods
MAX_ITER = 1000           # Maximum iterations for numerical methods
# Simulation methods
STEADYSTATE_METHOD="steadystate" 
INTEGRATION_METHOD="trapezoid"

# Performance boolean
PERFORMANCE = False # Set to True for performance testing, False for debugging