# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import importlib
class System:
    def __init__(self, models_list):
        self.models_list = models_list
        self.models = {}
        self.components = {}


    def import_models(self):
        for model_name in self.models_list:
            the_module = importlib.import_module('models.' + model_name)
            the_class = getattr(the_module, model_name)
            self.models[model_name] = the_class

    def add_components(self, model_name, component_info):
        self.import_models()
        model = self.models[model_name](params = component_info)
        self.components[model_name+component_info['idx']] = model









