# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

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

INITIAL_CONDITIONS = {'Bus':{'a':[0, 1, 2],
                             'v':[3, 4, 5]},
                      'ACLine':{'a1':[0, 1, 0],
                                'a2':[1, 2, 2],
                                'v1':[3, 4, 3],
                                'v2':[4, 5, 5]}}

# INITIAL_CONDITIONS = {'Bus':{'a':[0, 1],
#                              'v':[2, 3]},
#                       'ACLine':{'a1':[4],
#                                 'a2':[5],
#                                 'v1':[6],
#                                 'v2':[7]}}
