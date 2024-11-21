# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
from GridCalEngine.Simulations.ATC.scaling_factors import compute_sensed_factors, sensed_scale_to_reference


def test_1():
    values = np.array([[10, -20, 30], [40, 50, -60]])
    target = np.array([[10, 10, 10], [15, 15, 15]])
    idx = np.array([0, 1, 2])

    factors = compute_sensed_factors(
        values=values,
        idx=idx)

    scaled = sensed_scale_to_reference(
        values=values,
        target=target,
        idx=idx,
        imbalance=None,
        decimals=None,
    )

    c1 = np.allclose(np.sum(factors, axis=1), 1, rtol=1e-6)
    c2 = np.allclose(target.sum(axis=1) - scaled.sum(axis=1), 0, rtol=1e-6)
    assert (c1 and c2)



