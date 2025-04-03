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
            # assign an index to every variable:
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
            if isinstance(elem, AliasAlgeb):
                self.model_storage.add_aliasalgebs(elem)
            if isinstance(elem, AliasState):
                self.model_storage.add_aliastats(elem)
            if isinstance(elem, NumDynParam):
                self.model_storage.add_numdynparam(elem)
            if isinstance(elem, IdxDynParam):
                self.model_storage.add_idxdynparam(elem)
            if isinstance(elem, ExtParam):
                self.model_storage.add_extparam(elem)

####################### TO CLEAN ################################

    def calc_local_jacs(self, input_values):
        jacobians = []
        pycode_path = get_pycode_path()
        pycode_module = importlib.import_module(pycode_path.replace("/", "."))
        pycode_code = getattr(pycode_module, self.name)
        jacobian_info = pycode_code.jacobian_info
        for i in range(self.n):
            local_jac = pycode_code.g_ia(*input_values[i])
            jacobians.append(local_jac)
        return jacobian_info, jacobians




         
