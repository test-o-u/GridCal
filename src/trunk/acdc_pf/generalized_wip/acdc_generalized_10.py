import os
from GridCalEngine.Simulations.PowerFlow.power_flow_worker import PowerFlowOptions
from GridCalEngine.Simulations.PowerFlow.power_flow_options import SolverType
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

def run_time_67bus():
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

import numpy as np
elapsed = run_time_67bus()
times = np.zeros((1))
for i in range(1):
    elapsed = run_time_67bus()
    times[i] = elapsed


