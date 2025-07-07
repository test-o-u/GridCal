# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
from typing import List, Any, Dict
from GridCalEngine.Utils.Symbolic.symbolic import Const


class Event:
    def __init__(self,
                 prop: Any | None = None,
                 time_step: int = 0.0,
                 value: float = 0.0):
        self._prop = prop
        self._time_step = time_step
        self._value = value

    @property
    def prop(self):
        return self._prop

    @property
    def value(self):
        return self._value

    @property
    def time_step(self):
        return self._time_step



class Events:
    def __init__(self, events: List[Event]):
        self._events_dict = None
        self._events = events
        

    def fill_events_dict(self)-> Dict[int, list[Any]]:
        self._events_dict: Dict[int, list[Any]] = {}
        for event in self._events:
            self._events_dict[event.time_step] = list([event.prop, event.value])
        return self._events_dict


