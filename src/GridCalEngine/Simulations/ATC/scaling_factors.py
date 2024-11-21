# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from typing import Tuple, Union
from GridCalEngine.basic_structures import Logger, Vec, IntVec, Mat
from GridCalEngine.enumerations import AvailableTransferMode


def sensed_scale_to_reference(
        values: Union[Vec, Mat],
        target: Union[Vec, Mat],
        imbalance: Union[Vec, Mat, None] = None,
        idx: IntVec = None,
        decimals: Union[int, None] = None,
        logger: Union[Logger, None] = None) -> Union[Vec, Mat]:
    """

    :param values: A vector or matrix with values to scale.
    :param target: A vector or matrix representing the target values to achieve, based on their aggregate sum.
    :param imbalance: A vector or matrix representing the desired imbalance between the result and target values.
    :param idx:  A vector containing the indices that will participate in the scaling of the values vector.
    :param decimals: Integer. Number of decimals to fit the output.
    :param logger: A logger object isntance.
    :return: scaled values
    """
    # Determine the axis for summation based on the dimensionality of the input.
    # If values is a 1D array (vector), use axis=0.
    # If values is a 2D array (matrix), use axis=1.
    axis = 0 if values.ndim == 1 else 1

    if imbalance is None:
        imbalance = np.zeros_like(target)

    # Calculate delta based on the dimensions of target
    delta = target.sum(axis=axis) - values.sum(axis=axis) + imbalance.sum(axis=axis)

    # Compute scaling factors
    factors = compute_sensed_factors(values=values, idx=idx)

    # Scale the power values
    if values.ndim == 1:
        result = values + factors * delta
    else:
        result = values + factors * delta[:, np.newaxis]

    # Round the result if decimals is specified
    if decimals is not None:
        result = np.round(result, decimals=decimals)

    # Check if the target has been achieved
    difference = target.sum(axis=axis) - result.sum(axis=axis)
    if not np.allclose(difference, 0, rtol=1e-6):
        if logger:
            logger.add_warning('Issue computing sensed scale to reference: target is not achieved')

    return result


def compute_sensed_factors(
        values: Union[Vec, Mat],
        idx: IntVec = None,
        logger: Union[Logger, None] = None) -> Vec:
    """
    Compute scaling factors that adjust values considering their direction (positive or negative).
    This ensures that both positive and negative values are scaled proportionally in the same direction.
    :param values: A vector or matrix with values to scale
    :param idx:  A vector containing the indices that will participate in the scaling of the values vector.
    :param logger: Optional. Logger object
    :return: A vector of scaling factors that adjust the initial values
    """

    # Determine the axis for summation based on the dimensionality of the input.
    # If values is a 1D array (vector), use axis=0.
    # If values is a 2D array (matrix), use axis=1.
    axis = 0 if values.ndim == 1 else 1

    # Considering indices mask
    if idx is None:
        values_reference = values
    else:
        mask = np.zeros_like(values, dtype=bool)
        if values.ndim == 1:
            mask[idx] = True
        else:
            mask[:, idx] = True

        # zero in mask positions, power otherwise.
        values_reference = np.where(mask, values, 0)

    # Calculate the contribution proportions by sense:
    #   The goal is to scale positive and negative values proportionally in the same direction.

    # Calculate total absolute values for normalization
    total_abs_power = np.sum(np.abs(values_reference), axis=axis, keepdims=True)

    # Filter positive and negative values. Same vectors length, set not matched values to zero.
    power_pos = np.where(values_reference < 0, 0, values_reference)
    power_neg = np.where(values_reference > 0, 0, values_reference)

    # Calculate total absolute values up/down for normalization
    total_power_pos = np.sum(power_pos, axis=axis, keepdims=True)
    total_power_neg = np.sum(power_neg, axis=axis, keepdims=True)
    total_abs_power_pos = np.sum(np.abs(power_pos), axis=axis, keepdims=True)
    total_abs_power_neg = np.sum(np.abs(power_neg), axis=axis, keepdims=True)

    # Calculate factor related to sense contribution
    factors_sense_up = np.divide(total_power_pos, total_abs_power, where=total_abs_power != 0)
    factors_sense_dw = np.divide(total_power_neg, total_abs_power, where=total_abs_power != 0)

    # Calculate factor related individual contribution
    factors_values_up = np.divide(power_pos, total_abs_power_pos, where=total_abs_power_pos != 0)
    factors_values_dw = np.divide(power_neg, total_abs_power_neg, where=total_abs_power_neg != 0)

    # Calculate combined factors
    factors_power_delta_up = factors_values_up * factors_sense_up
    factors_power_delta_dw = factors_values_dw * factors_sense_dw

    # Join factors
    # Note: This is not a summation; it's simply combining values using a logical 'or' operation.
    factors = factors_power_delta_up + factors_power_delta_dw

    # Check if the sum of factors is close to 1
    if not np.allclose(np.sum(factors, axis=axis), 1, rtol=1e-6):
        if logger:
            logger.add_warning('Issue computing sensed factors, factors sum is not close to one.')

    return factors


def compute_exchange_factors(
        values: Vec,
        up_idx: IntVec,
        down_idx: IntVec,
        logger: Logger):
    """
    Compute value factors by transfer method with sign consideration.
    :param values: A vector or matrix with values to scale
    :param up_idx: A vector containing the indices that will participate in the scaling up of the values vector.
    :param down_idx: A vector containing the indices that will participate in the scaling down of the values vector.
    :param logger: A logger object instance
    :return: A vector of scaling factors that adjust the initial values
    """
    factors_up = compute_sensed_factors(values=values, idx=up_idx, logger=logger)
    factors_down = compute_sensed_factors(values=values, idx=down_idx, logger=logger)

    factors = factors_up - factors_down

    return factors


def compute_nodal_power_by_transfer_method(
        generation_per_bus: Vec,
        load_per_bus: Vec,
        pmax_per_bus: Vec,
        transfer_method: AvailableTransferMode) -> Vec:
    """
    Returns nodal power according to the transfer_method.
    :param generation_per_bus: Generation per bus
    :param load_per_bus: Load per bus
    :param pmax_per_bus: Pmax per bus
    :param transfer_method: Exchange transfer method
    :return: nodal power (p.u.)
    """

    # Evaluate transfer method
    if transfer_method == AvailableTransferMode.InstalledPower:
        p_ref = pmax_per_bus

    elif transfer_method == AvailableTransferMode.Generation:
        p_ref = generation_per_bus

    elif transfer_method == AvailableTransferMode.Load:
        p_ref = load_per_bus

    elif transfer_method == AvailableTransferMode.GenerationAndLoad:
        p_ref = generation_per_bus - load_per_bus

    else:
        raise Exception('Undefined available transfer mode')

    return p_ref


def compute_nodal_max_power_by_transfer_method(
        pmax_per_bus: Vec,
        transfer_method: AvailableTransferMode,
        skip_generation_limits: bool,
        inf_value: float) -> Vec:
    """
    Returns nodal max power according to the transfer_method.
    :param pmax_per_bus: Pmax per bus
    :param transfer_method: Exchange transfer method
    :param skip_generation_limits: Skip generation limits?
    :param inf_value: infinity value. Ex 1e-20
    :return: nodal max power (p.u.)
    """

    nbus = len(pmax_per_bus)

    # Evaluate transfer method
    if transfer_method == AvailableTransferMode.InstalledPower:
        p_max = pmax_per_bus

    elif transfer_method == AvailableTransferMode.Generation:
        p_max = pmax_per_bus

    elif transfer_method == AvailableTransferMode.Load:
        p_max = np.full(nbus, inf_value)

    elif transfer_method == AvailableTransferMode.GenerationAndLoad:
        p_max = np.full(nbus, inf_value)

    else:
        raise Exception('Undefined available transfer mode')

    if skip_generation_limits:
        p_max = np.full(nbus, inf_value)

    return p_max


def compute_nodal_min_power_by_transfer_method(
        pmin_per_bus: Vec,
        transfer_method: AvailableTransferMode,
        skip_generation_limits: bool,
        inf_value: float) -> Vec:
    """
    Returns nodal pmin according to the transfer_method.
    :param pmin_per_bus: Pmin per bus
    :param transfer_method: Exchange transfer method
    :param skip_generation_limits: Skip generation limits?
    :param inf_value: infinity value. Ex 1e-20
    :return: nodal min power
    """

    nbus = len(pmin_per_bus)

    # Evaluate transfer method
    if transfer_method == AvailableTransferMode.InstalledPower:
        p_min = pmin_per_bus

    elif transfer_method == AvailableTransferMode.Generation:
        p_min = pmin_per_bus

    elif transfer_method == AvailableTransferMode.Load:
        p_min = np.full(nbus, -inf_value)

    elif transfer_method == AvailableTransferMode.GenerationAndLoad:
        p_min = np.full(nbus, -inf_value)

    else:
        raise Exception('Undefined available transfer mode')

    if skip_generation_limits:
        p_min = np.full(nbus, -inf_value)

    return p_min
