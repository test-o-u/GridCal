# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import os
import importlib
import compileall
import time  
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Devices.Dynamic.dae import DAE
from GridCalEngine.Devices.Dynamic.utils.paths import get_pycode_path
from GridCalEngine.Devices.Dynamic.io.json import readjson


class System:
    "This class contains the models and devices."

    def __init__(self, models_list, datafile):

        self.models_list = models_list

        self.models = {}
        self.devices = {}

        self.dae = DAE()

        self.data = readjson(datafile)

        self.import_models()
        self.system_prepare()


    def import_models(self):
        "This function imports all the models, stores its information in a spoint object, does the symbolic-numeric transformation and stores the numeric code in .py file."
        for model_type, model_names in self.models_list:
            for model_name in model_names:
                model_module = importlib.import_module('GridCalEngine.Devices.Dynamic.models.' + model_type)
                cls = getattr(model_module, model_name)
                model = cls(name=model_name, code='', idtag='')
                self.models[model_name] = model

    def system_prepare(self):

        # 1. Process models symbolically and lambdify functions (generate fast numerical functions)
        symb_st = time.perf_counter()

        for model in self.models.values():
            model.process_symbolic()
            # model.store_data()  
        self.finalize_pycode()

        symb_end = time.perf_counter() 
        self.symb_time = symb_end - symb_st

        # 2. Create a model instance per device type to store multiple devices in a vectorized form
        dev_st = time.perf_counter()

        self.create_devices(self.data)

        dev_end = time.perf_counter() 
        self.dev_time = dev_end - dev_st

        # 3. Assign global indexes to variables and copy of global indexes to external variables 
        add_st = time.perf_counter()

        self.set_addresses()

        add_end = time.perf_counter() 
        self.add_time = add_end - add_st

    def create_devices(self, data):
        for model_name, model_data in data.items():
            model = self.models[model_name]
            for device in model_data:
                model.n += 1
                for name, val in device.items():
                    if hasattr(model, name):
                        param = getattr(model, name)
                        if isinstance(param, IdxDynParam):
                            param.id.append(val)
                        if isinstance(param, NumDynParam):
                            param.value.append(val)

    def finalize_pycode(self):
        pycode_path = get_pycode_path()
        init_path = os.path.join(pycode_path, '__init__.py')
        with open(init_path, 'w') as f:
            for name in self.models.keys():
                f.write(f"from . import {name}  \n")
            f.write('\n')
        compileall.compile_dir(pycode_path)

    def set_addresses(self):
        self.global_id = 0
        algeb_ref_map = {}  # Cache: store algeb_idx references for quick lookup

        # First loop: Process AlgebVar
        for model_instance in self.models.values():
            for var_list in model_instance.__dict__.values():
                if isinstance(var_list, AlgebVar):
                    indices = list(range(self.global_id, self.global_id + model_instance.n))
                    model_instance.algeb_idx[var_list.name] = indices
                    self.global_id += model_instance.n  # Move global index forward

                    # Cache reference for faster lookup
                    algeb_ref_map[(model_instance.__class__.__name__, var_list.name)] = indices

        # Second loop: Process ExternAlgeb
        for model_instance in self.models.values():
            for var_list in model_instance.__dict__.values():
                if isinstance(var_list, ExternAlgeb):
                    key = (var_list.indexer.symbol, var_list.src)

                    if key not in algeb_ref_map:
                        raise KeyError(f"Variable '{var_list.src}' not found in {var_list.indexer.symbol}.algeb_idx")

                    parent_idx = algeb_ref_map[key]  # Retrieve index from cache

                    # Store in extalgeb_idx using src as the key (grouping multiple references)
                    if var_list.src not in model_instance.extalgeb_idx:
                        model_instance.extalgeb_idx[var_list.src] = []  # Initialize as a list of lists

                    model_instance.extalgeb_idx[var_list.src].append([parent_idx[i] for i in var_list.indexer.id])

    def update_jacobian(self):
        all_triplets = {}
        # for model in self.models.values():
        model = self.models['ACLine']

        # get the function type, var type info and the local jacobians
        jacobian_info, local_jacobians = model.calc_local_jacs(model)
        # print(local_jacobians[])
        var_adresses = {0: ('a1', (0, 1, 2)),
                        1: ('g21', (3, 4, 5)),
                        2: ('a2', (6, 7, 8)),
                        3: ('u', (9, 10, 11)),
                        4: ('v1', (12, 13, 14)),
                        5: ('b', (15, 16, 17)),
                        6: ('g', (18, 19, 20)),
                        7: ('v2', (21, 22, 23)),
                        8: ('bsh', (24, 25, 26)),
                        9: ('b21', (27, 28, 29))}
        for jac_type, positions in zip(jacobian_info.keys(), jacobian_info.values()):
            if jac_type == 'dgy':
                triplets = self.assign_positions(model.n, local_jacobians, jac_type, positions, var_adresses)
                all_triplets[jac_type] = triplets


    def assign_positions(self, num_components, local_jacobian, jac_type, positions, var_adresses):
        triplets = []
        i = 0
        while i < num_components:
            j = 0
            for elem in positions:
                val = local_jacobian[i][j]
                func_index, var_index = elem
                adress_func = var_adresses[func_index][1][i]
                adress_var = var_adresses[var_index][1][i]
                triplet = (adress_func, adress_var, val)
                triplets.append(triplet)
                j += 1
            i += 1
        return triplets
