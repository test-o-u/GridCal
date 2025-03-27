# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import os
import importlib
import compileall
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.Devices.Dynamic.utils.paths import get_pycode_path
from GridCalEngine.Devices.Dynamic.dae import DAE

class System:
    "This class contains the models and devices."

    def __init__(self, models_list):
        self.models_list = models_list
        self.models = {}
        self.devices = {}
        self.dae = DAE()

    def import_models(self):
        "This function imports all the models, stores its information in a spoint object, does the symbolic-numeric transformation and stores the numeric code in .py file."
        for model_type, models in self.models_list:
            for model in models:
                the_module = importlib.import_module('GridCalEngine.Devices.Dynamic.models.' + model_type)
                the_class = getattr(the_module, model)
                model_object = the_class(name=model, code='', idtag='')
                self.models[model] = model_object

    def prepare(self, components_info):
        for model in self.models.values():
            model.store_data()
            model.process_data()
        self.finalize_pycode()
        self.create_devices(components_info)
        self.set_addresses()
        
        print(f"Bus a = {self.models['Bus'].algeb_idx['a']}")
        # print(f"Bus v = {self.models['Bus'].algeb_idx['v']}")
        print(f"ACLine a = {self.models['ACLine'].extalgeb_idx['a']}")
        # print(f"ACLine1 v1 = {self.models['ACLine'].extalgeb_idx['v1']}")
        # print(f"ACLine a2 = {self.models['ACLine'].extalgeb_idx['a2']}")
        # print(f"ACLine2 v2 = {self.models['ACLine'].extalgeb_idx['v2']}")
        # print(f"ACLine3 a3 = {self.models['ACLine'].extalgeb_idx['a2']}")
        # print(f"ACLine3 v3 = {self.models['ACLine'].extalgeb_idx['v2']}")

    def create_devices(self, components_info):
        for model_name, model_entries in components_info.items():
            model = self.models[model_name]
            for entry in model_entries:
                model.n += 1 
                for param_name, val in entry.items():
                    if hasattr(model, param_name):
                        param = getattr(model, param_name)
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
    
    





                        
                        
                        













    # def prepare(self, components_info):
    #   for model_name, dct in components_info.items():
    #      for component_info in dct:
    #         self.add_component(model_name, component_info)

    # 