from pygent.common import PygentOperator
from pygent.context import BaseContext
from pygent.module import PygentModule


class BaseAgent(PygentOperator):
    def __init__(self, *args, **kwargs):
        super().__init__()

    async def forward(self, user_input: str):
        pass
