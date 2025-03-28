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
    ('line', ['ACLine']), 
    ('bus', ['Bus']),
    # ('load', ['ExpLoad']) 
    # ('syngen', ['GENCLS'])
])

INITIAL_CONDITIONS = {'Bus 1':[],
                      'Bus 2':[],
                      'Bus 3':[],
                      'ACLine 1':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'ACLine 2':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'ACLine 3':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}