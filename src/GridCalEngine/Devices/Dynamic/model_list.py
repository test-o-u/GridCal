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
    # ('load', ['ExpLoad']), 
    # ('syngen', ['GENCLS'])
])          