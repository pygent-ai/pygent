from typing import List

from pygent.common import PygentOperator, PygentString, PygentList
from pygent.message import BaseMessage, SystemMessage


class BaseContext(PygentOperator):
    history: PygentList[BaseMessage]

    def __init__(self, system_prompt=None, *args, **kwargs) -> None:
        super().__init__()
        self.system_prompt = None if system_prompt is None else PygentString(system_prompt)
        if system_prompt is not None:
            self.history = PygentList([SystemMessage(self.system_prompt)])

    def add_message(self, message: BaseMessage):
        self.history.append(message)

    @property
    def last_message(self):
        return self.history[-1]

