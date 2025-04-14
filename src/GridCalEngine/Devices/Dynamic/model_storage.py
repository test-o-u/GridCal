# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from collections import defaultdict
class ModelStorage:
    """
    A class to store and manage model expressions and variables (string form).

    Attributes:
        name (str): The name of the model component.
        f (list): List of differential equations (state equations).
        g (list): List of algebraic equations.
        all_vars (list): Collection of all variables in the model.
        stats (list): List of state variables.
        algebs (list): List of algebraic variables.
        stats_symb (list): List of symbols representing state variables.
        algebs_symb (list): List of symbols representing algebraic variables.
        statVars (list): State variables (differential).
        algebVars (list): Algebraic variables.
        externVars (list): External input variables.
        externStates (list): External state variables.
        externAlgebs (list): External algebraic variables.
        aliasAlgebs (list): Alias algebraic variables (for reference).
        aliasStats (list): Alias state variables (for reference).
        numdynParam (list): Numeric dynamic parameters.
        idxdynParam (list): Indexed dynamic parameters.
        extdynParam (list): External dynamic parameters.
    """

    def __init__(self, model_name):
        """
        Initialize a model component with a given name.
        """
        self.name = model_name

        # Lists to store model equations

        # Collections of variables
        self.stats = []
        self.algebs = []

        # Categorized variable types

        # Parameters
        self.numdynParam = []
        self.idxdynParam = []


    def add_statvars(self, expr):
        """
        Add a state variable (differential equation).
        """
        self.stats.append(expr)

    def add_algebvars(self, expr):
        """
        Add an algebraic variable (algebraic equation).
        """
        self.algebs.append(expr)

    def add_externstates(self, expr):
        """
        Add an external state variable (state variable defined externally).
        """
        self.stats.append(expr)

    def add_externalgebs(self, expr):
        """
        Add an external algebraic variable (algebraic variable defined externally).
        """
        self.algebs.append(expr)





