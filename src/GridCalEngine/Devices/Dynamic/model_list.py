# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import numpy as np
"""
This module defines the power system models organized in categories.

Constants:
    MODELS (list): A list of tuples mapping model categories to their respective models.
"""

MODELS = list([
    ('bus', ['Bus']),
    ('line', ['ACLine']), 
    ('load', ['ExpLoad']),
    ('syngen', ['GENCLS'])
])

x0 = {
    'delta': 0.0,
    'omega': 1.0,
}

y0 = {
    'a1': 15 * (np.pi / 180),  # rotor angle (rad)
    'a2': 10 * (np.pi / 180),  # angle of second bus, possibly infinite bus
    'v1': 1.0,                # generator terminal voltage magnitude (pu)
    'v2': 0.95,                 # remote bus voltage (pu)

    # Stator dq axis flux linkages (GENCLS has no field circuit, so usually derived from voltage and current)
    'psid': 1.0,               # flux linkage in d-axis (pu)
    'psiq': 0.0,               # flux linkage in q-axis (pu)

    # dq axis currents — assuming steady state power flow, estimated from power and voltage
    'i_d': 0.1,                # d-axis stator current (pu)
    'i_q': 0.2,                # q-axis stator current (pu)

    # dq terminal voltages (transformed from v1, a1)
    'vd': 0.0,                 # d-axis voltage (pu)
    'vq': 1.0,                # q-axis voltage (pu)
    
    # Electromagnetic torque
    'te': 0.2,                 # electric torque (pu)

    # Electrical power output
    'Pe': 0.2,                 # real power (pu)
    'Qe': 0.2,                 # reactive power (pu)
}

dae_x0 = np.array(list(x0.values()))
dae_y0 = np.array(list(y0.values()))
