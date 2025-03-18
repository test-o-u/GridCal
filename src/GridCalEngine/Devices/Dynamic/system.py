# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import importlib
from GridCalEngine.Utils.dyn_param import NumDynParam
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate


class System:
    "This class contains the models and components."

    def __init__(self, models_list):
        self.models_list = models_list
        self.models = {}
        self.components = []

    def import_models(self):
        "this function imports all the models, stores its information in a spoint object, does the symbolic-numeric transformation and stores the numeric code in .py file."
        for model_type, devices in self.models_list:
            for device in devices:
                the_module = importlib.import_module('GridCalEngine.Devices.Dynamic.models.' + model_type)
                the_class = getattr(the_module, device)
                self.models[device] = the_class
                the_class.store_data()
                the_class.process_data()

    def prepare(self):
        self.import_models()
















    # def prepare(self, components_info):
    #   for model_name, dct in components_info.items():
    #      for component_info in dct:
    #         self.add_component(model_name, component_info)

    # def add_component(self, model_name, component_info):
    #   "this function creates a component, updats the value of its parameters using the information coming from the json file and stores it into self.components"
    #  component = self.models[model_name](name=model_name, code=component_info['code'], idtag='')
    # for key, val in list(component_info.items()):
    #    element = getattr(component, key)
    #   if isinstance(element, NumDynParam):
    #      element.value = val
    # self.components.append(component)
    # store function from dynamic_madel_template, it stores the component info into the object spoint
    # component.store_data()
    # component.process_data()
