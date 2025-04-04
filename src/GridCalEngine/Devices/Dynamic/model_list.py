# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import numpy as np
"""
This module defines the power system models organized in categories.

Constants:
    MODELS (list): A list of tuples mapping model categories to their respective models.
"""

MODELS = list([
    ('bus', ['Bus']),
    ('line', ['ACLine']), 
    ('load', ['ExpLoad']),
    ('syngen', ['GENCLS'])
])

#INITIAL_CONDITIONS = {'Bus':{'a':[0, 1, 2],
 #                            'v':[3, 4, 5]},
  #                    'ACLine':{'a1':[0, 1, 0],
   #                             'a2':[1, 2, 2],
    #                            'v1':[3, 4, 3],
     #                           'v2':[4, 5, 5]}}



# INITIAL_CONDITIONS = {'Bus':{'a':[0, 1],
#                              'v':[2, 3]},
#                       'ACLine':{'a1':[4],
#                                 'a2':[5],
#                                 'v1':[6],
#                                 'v2':[7]}}

INITIAL_CONDITIONS = {'Bus':{'a':[0, 1],
                             'v':[2, 3]},
                      'ACLine':{'a1':[0],
                                'a2':[1],
                                'v1':[2],
                                'v2':[3]},
                      'GENCLS':{'delta':[4],
                                'omega':[5],
                                'psid':[6],
                                'psiq':[7],
                                'i_d':[8],
                                'i_q':[9],
                                'vd':[10],
                                'vq':[11],
                                'te':[12],
                                'Pe':[13],
                                'Qe':[14],
                                'a':[0],
                                'v':[2]},
                      'ExpLoad':{'a':[1],
                                 'v':[3]}}



#DAEY = np.array([0, 1, 0, 1, 2, 2, 3, 4, 3, 4, 5, 5 ])
DAEX = np.array([])
DAEXY = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 2, 1, 3])
