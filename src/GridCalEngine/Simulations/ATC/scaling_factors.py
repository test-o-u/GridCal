
import numpy as np
from typing import Tuple, Union
from GridCalEngine.basic_structures import Logger, Vec, IntVec
from GridCalEngine.enumerations import AvailableTransferMode


def compute_sensed_factors(
        power: Vec,
        idx: IntVec,
        logger: Union[Logger, None] = None) -> Vec:
    """
    Compute sensed factors to allow scaling considering sense of power (increase positive and decrease negative accordingly)
    :param power: full power values to scale
    :param idx: value indices to consider in scaling
    :param logger: logger
    :return: scale factors
    """
    nelem = len(power)

    # bus area mask
    isin_ = np.isin(range(nelem), idx, assume_unique=True)

    power_reference = power * isin_

    # get proportions of contribution by sense (gen or pump) and area
    # the idea is both techs contributes to achieve the power shift goal in the same proportion
    # that in base situation

    # Filter positive and negative power. Same vectors length, set not matched values to zero.
    power_pos = np.where(power_reference < 0, 0, power_reference)
    power_neg = np.where(power_reference > 0, 0, power_reference)

    factors_up = np.sum(power_pos) / np.sum(np.abs(power_reference))
    factors_dw = np.sum(power_neg) / np.sum(np.abs(power_reference))

    # get proportion by production (amount of power contributed by generator to his sensed area).
    if np.sum(np.abs(power_pos)) != 0:
        factors_up_power = power_pos / np.sum(np.abs(power_pos))
    else:
        factors_up_power = np.zeros_like(power_pos)

    if np.sum(np.abs(power_neg)) != 0:
        factors_dw_power = power_neg / np.sum(np.abs(power_neg))
    else:
        factors_dw_power = np.zeros_like(power_neg)

    # delta factors by power (considering both factors: sense and power)
    factors_power_delta_up = factors_up_power * factors_up
    factors_power_delta_dw = factors_dw_power * factors_dw

    # Join power factors into one vector
    # Notice this is not a summatory, it's just joining like 'or' logical operation
    factors = factors_power_delta_up + factors_power_delta_dw

    # some checks
    if not np.isclose(np.sum(factors), 1, rtol=1e-6):
        if logger:
            logger.add_warning('Issue computing factors to scale delta power in area.')

    return factors


def compute_exchange_factors(
        power: Vec,
        up_idx: IntVec,
        down_idx: IntVec,
        logger: Logger):
    """
    Get generation factors by transfer method with sign consideration.
    :param power: Vec. Power reference
    :param up_idx: indices to increase
    :param down_idx: indices to decrease
    :param logger: logger instance
    :return: factors, sense, p_max, p_min
    """
    factors_up = compute_sensed_factors(power=power, idx=up_idx, logger=logger)
    factors_down = compute_sensed_factors(power=power, idx=down_idx, logger=logger)

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
    :return: nodal power (p.u.)
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
    Returns nodal power, pmax and pmin according to the transfer_method.
    :param pmin_per_bus: Pmin per bus
    :param transfer_method: Exchange transfer method
    :param skip_generation_limits: Skip generation limits?
    :param inf_value: infinity value. Ex 1e-20
    :return: nodal power (p.u.)
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