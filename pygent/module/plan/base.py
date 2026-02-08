from typing import Any

from pygent.module import PygentModule
from pygent.context import BaseContext


class BasePlan(PygentModule):
    todo_list: Any

    def __init__(self, *args, **kwargs):
        super().__init__()


    def forward(self, context: BaseContext):
        pass


