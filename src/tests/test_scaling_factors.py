# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
from GridCalEngine.Simulations.ATC.scaling_factors import compute_sensed_factors, sensed_scale_to_reference

def test_1():
    """
    Test compute_sensed_factors with vectors
    :return:
    """
    vector = np.array([10, -20, 30])
    idx = np.array([0, 1, 2])

    factors_vector = compute_sensed_factors(
        values=vector,
        idx=idx)

    assert (np.allclose(np.sum(factors_vector, axis=0), 1, rtol=1e-6))

def test_2():
    """
    Test compute_sensed_factors with matrix
    :return:
    """
    matrix = np.array([[10, -20, 30], [40, 50, -60]])
    idx = np.array([0, 1, 2])

    factors_matrix = compute_sensed_factors(
        values=matrix,
        idx=idx)

    assert (np.allclose(np.sum(factors_matrix, axis=1), 1, rtol=1e-6))


def test_3():
    """
    Test sensed_scale_to_reference function with vectors
    :return:
    """

    values_vector = np.array([10, -20, 30])
    target_vector = np.array([10, 10, 10])

    idx = np.array([0, 1, 2])

    scaled_vector = sensed_scale_to_reference(
        values=values_vector,
        target=target_vector,
        idx=idx,
        imbalance=None,
        decimals=None,
    )

    c1 = np.allclose(np.sum(scaled_vector, axis=0) - np.sum(target_vector, axis=0), 0, rtol=1e-6)
    assert (c1)


def test_4():
    """
    Test sensed_scale_to_reference function with matrix
    :return:
    """
    values_matrix = np.array([[10, -20, 30], [40, 50, -60]])
    target_matrix = np.array([[10, 10, 10], [15, 15, 15]])

    idx = np.array([0, 1, 2])

    scaled_matrix = sensed_scale_to_reference(
        values=values_matrix,
        target=target_matrix,
        idx=idx,
        imbalance=None,
        decimals=None,
    )

    assert (np.allclose(np.sum(scaled_matrix, axis=1) - np.sum(target_matrix, axis=1), 0, rtol=1e-6))
