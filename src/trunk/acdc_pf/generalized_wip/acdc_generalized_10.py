import os
from GridCalEngine.Simulations.PowerFlow.power_flow_worker import PowerFlowOptions
from GridCalEngine.Simulations.PowerFlow.power_flow_options import SolverType
from GridCalEngine.enumerations import ConverterControlType
import GridCalEngine.api as gce
import faulthandler

faulthandler.enable()  # start @ the beginning
def run_time_5bus():
    """
    Check that a transformer can regulate the voltage at a bus
    """
    fname = os.path.abspath("C:/Users/raiya/Documents/8. eRoots/HVDCPAPER/leuvenTestCasesACDC/case5_3_he_fixed_controls_final.gridcal")
    # fname = "G:/.shortcut-targets-by-id/1B4zzyZBFXXFuEGTYGLt-sPLVc6VD2iL4/eRoots Analytics Shared Drive/Development/Project ACDC1 AC-DC Power Flow/Training grids/5714v2.gridcal"

    grid = gce.open_file(fname)

    options = PowerFlowOptions(SolverType.NR,
                               verbose=2,
                               control_q=False,
                               retry_with_other_methods=False,
                               control_taps_phase=True,
                               control_taps_modules=True,
                               max_iter=80,
                               tolerance=1e-8, )

    results = gce.power_flow(grid, options)

    print(results.get_bus_df())
    # print(results.get_branch_df())
    # print("results.error", results.error)
    print("results.elapsed_time", results.elapsed)
    return results.elapsed
    # assert results.converged

def run_time_39bus():
    """
    Check that a transformer can regulate the voltage at a bus
    """
    fname = os.path.abspath("C:/Users/raiya/Documents/8. eRoots/HVDCPAPER/leuvenTestCasesACDC/case24_7_jb.gridcal")

    grid = gce.open_file(fname)

    # for j in range(len(grid.vsc_devices)):
    #     print(grid.vsc_devices[j].name)
    #     print("control1:", grid.vsc_devices[j].control1)
    #     print("control1val:", grid.vsc_devices[j].control1_val)
    #     print("control2:", grid.vsc_devices[j].control2)
    #     print("control2val:", grid.vsc_devices[j].control2_val)
    grid.vsc_devices[0].control1 = ConverterControlType.Vm_dc
    grid.vsc_devices[1].control1 = ConverterControlType.Pac
    grid.vsc_devices[2].control1 = ConverterControlType.Pac
    grid.vsc_devices[3].control1 = ConverterControlType.Vm_dc
    grid.vsc_devices[4].control1 = ConverterControlType.Pac
    grid.vsc_devices[5].control1 = ConverterControlType.Pac
    grid.vsc_devices[6].control1 = ConverterControlType.Pac


    options = PowerFlowOptions(SolverType.NR,
                               verbose=2,
                               control_q=False,
                               retry_with_other_methods=False,
                               control_taps_phase=True,
                               control_taps_modules=True,
                               max_iter=80,
                               tolerance=1e-8, )

    results = gce.power_flow(grid, options)

    print(results.get_bus_df())
    # print(results.get_branch_df())
    # print("results.error", results.error)
    print("results.elapsed_time", results.elapsed)
    return results.elapsed
    # assert results.converged

import numpy as np
elapsed = run_time_39bus()
times = np.zeros((1))
for i in range(1):
    elapsed = run_time_39bus()
    times[i] = elapsed


