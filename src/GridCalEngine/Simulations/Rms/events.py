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


class ConstEvent(Event):
    def __init__(self, prop: Const | None = None, time_step: int = 0.0, value=0.0):
        """

        :param constant:
        :param idtag:
        :param name:
        :param code:
        :param time_step:
        :param value:
        """

        super().__init__(prop, time_step, value)
        self._time_step: int = time_step
        self._prop: Const = prop
        self._value = value

class Events:
    def __init__(self, events: List[Event]):
        self._events = events
        self._events_dict: Dict[int, list[Any]] = {}
        self.build_events_dict()

    def build_events_dict(self):
        for event in self._events:
            events_dict = self.get_events_dict()
            events_dict[event.time_step] = list([event.prop, event.value])

    def get_events_dict(self) -> Dict[int, list[Any]]:
        return self._events_dict


