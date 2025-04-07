# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import importlib
import numpy as np
from GridCalEngine.Devices.Parents.editable_device import EditableDevice, DeviceType
from typing import Union
from GridCalEngine.Devices.Dynamic.utils.paths import get_pycode_path
from GridCalEngine.Devices.Dynamic.model_storage import ModelStorage
from GridCalEngine.Devices.Dynamic.symprocess import SymProcess
from GridCalEngine.Utils.dyn_param import NumDynParam
from GridCalEngine.Utils.dyn_var import *
from GridCalEngine.Utils.dyn_param import *



class DynamicModelTemplate(EditableDevice):
    """
    Represents a dynamic model template for a device, handling symbolic processing,
    storage of variables, and setting addresses.

    Inherits from EditableDevice, allowing dynamic model creation and symbolic processing.
    """
    def __init__(self, name: str, code: str, idtag: Union[str, None],
                 device_type: DeviceType):
        """
        Initializes a dynamic model template with symbolic processing and storage.

        :param name: Name of the dynamic model.
        :param code: Unique code identifier.
        :param idtag: Optional tag for identifying the model instance.
        :param device_type: The type of the device (e.g., generator, load, etc.).
        """
        EditableDevice.__init__(self,
                                name=name,
                                code=code,
                                idtag=idtag,
                                device_type=device_type)

        # Storage for model variables and parameters
        self.model_storage = ModelStorage(self.name)

        # Dictionary containing instance attributes
        self.dict = self.__dict__
        
        # Symbolic processing engine
        self.sym = SymProcess(self)

        # dictionary containing index of the variable as key and symbol of the variable as value
        self.vars_index = {}

        # list containing all the symbols of the variables in the model
        self.variables_list = []

        # Address mapping for algebraic variables

        # Set address function
        self.n = 0
        self.states_idx = {}
        self.extstates_idx = {}
        self.algeb_idx = {}     # Dictionary for algebraic variable indexing
        self.extalgeb_idx = {}  # Dictionary for external algebraic variable indexing
        
    def process_symbolic(self):
        """
        Generates symbolic equations and Jacobians for the dynamic model.
        """
        self.sym.generate()

    def store_data(self):
        """
        Stores different types of variables and parameters in the model storage.
        This method categorizes each instance variable and adds it to the corresponding 
        storage structure.

        Also, it saves a list with all the variables of a model and creates a dictionary with an index as key and the variable name as value
        """

        index = 0
        for key, elem in self.dict.items():
            # assign an index to every variable in the model populating vars_index dictionary
            if isinstance(elem, DynVar):
                self.variables_list.append(elem.symbol)
                self.vars_index[index] = elem.symbol
                index += 1

            if isinstance(elem, AlgebVar):
                self.model_storage.add_algebvars(elem)
            if isinstance(elem, StatVar):
                self.model_storage.add_statvars(elem)
            if isinstance(elem, ExternVar):
                self.model_storage.add_externvars(elem)
            if isinstance(elem, ExternState):
                self.model_storage.add_externstates(elem)
            if isinstance(elem, ExternAlgeb):
                self.model_storage.add_externalgebs(elem)

            #if isinstance(elem, NumDynParam):
             #   self.dae.params_dict[self.name][elem.symbol] = elem.value
            #if isinstance(elem, IdxDynParam):
             #   self.model_storage.add_idxdynparam(elem)
            #if isinstance(elem, ExtParam):
             #   self.model_storage.add_extparam(elem)

####################### TO CLEAN ################################
    def import_pycode(self):
        pycode_path = get_pycode_path()
        pycode_module = importlib.import_module(pycode_path.replace("/", "."))
        pycode_code = getattr(pycode_module, self.name)

        return pycode_code

    def calc_f_g_functions(self, f_input_values, g_input_values):
        pycode_code = self.import_pycode()
        f_values_device = []
        g_values_device = []

        for i in range(self.n):
            # get f values
            if f_input_values:
                f_values = pycode_code.f_update(*f_input_values[i])
                f_values_device.append(f_values)
            #get g values
            if g_input_values:
                g_values = pycode_code.g_update(*g_input_values[i])
                g_values_device.append(g_values)
        f_values_device_flat = [val for component in f_values_device for val in component]
        g_values_device_flat = [val for component in g_values_device for val in component]
        return f_values_device_flat, g_values_device_flat

    def calc_local_jacs(self, f_input_values, g_input_values):
        pycode_code = self.import_pycode()
        jacobian_info = pycode_code.jacobian_info
        f_jacobians = []
        g_jacobians = []
        for i in range(self.n):
            if f_input_values:
                local_jac_f = pycode_code.f_ia(*f_input_values[i])
                f_jacobians.append(local_jac_f)
            if g_input_values:
                local_jac_g = pycode_code.g_ia(*g_input_values[i])
                g_jacobians.append(local_jac_g)
        return f_jacobians, g_jacobians, jacobian_info
            




         
