# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import os
import importlib
import compileall
from GridCalEngine.Utils.dyn_param import NumDynParam
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.Devices.Dynamic.utils.paths import get_pycode_path

class System:
    "This class contains the models and components."

    def __init__(self, models_list):
        self.models_list = models_list
        self.models = {}
        self.components = []

    def import_models(self):
        "this function imports all the models, stores its information in a spoint object, does the symbolic-numeric transformation and stores the numeric code in .py file."
        for model_type, models in self.models_list:
            for model_name in models:
                the_module = importlib.import_module('GridCalEngine.Devices.Dynamic.models.' + model_type)
                the_class = getattr(the_module, model_name)
                model = the_class(name=model_name, code='', idtag='')
                self.models[model_name] = model


    def prepare(self):
        self.import_models()
        for model in self.models.values():
            model.store_data()
            model.process_data()
        self.finalize_pycode()


    def finalize_pycode(self):
        pycode_path = get_pycode_path()
        init_path = os.path.join(pycode_path, '__init__.py')
        with open(init_path, 'w') as f:
           #f.write(f"__version__ = '{DynamicSimulator.__version__}'\n\n")

            for name in self.models.keys():
                f.write(f"from . import {name}  \n")
            f.write('\n')
        compileall.compile_dir(pycode_path)

        return
