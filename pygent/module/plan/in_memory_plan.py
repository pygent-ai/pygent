from typing import List, Dict, Any

from pygent.module.tool import BaseTool
from pygent.common import PygentOperator, PygentList, PygentString, PygentInt


class PygentStatus:
    """Status enum for todo items (values match PygentEnum-style)."""
    PENDING = 0
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3


class InMemoryTodoItem(PygentOperator):
    """Single todo item: content (text) and status."""
    content: PygentString
    status: PygentInt

    def __init__(self, content: str = "", status: int = PygentStatus.PENDING, **kwargs):
        super().__init__()
        self.content.data = content
        self.status.data = status

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content.data, "status": self.status.data}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InMemoryTodoItem":
        return cls(content=d.get("content", ""), status=d.get("status", PygentStatus.PENDING))


class _CreateTodoListTool(BaseTool):
    """Tool that creates a new todo list, replacing any existing one."""

    def __init__(self, plan: "InMemoryPlan"):
        super().__init__(
            name="create_todo_list",
            description="Create a new todo list, replacing any existing one. Pass a list of task description strings.",
        )
        self._plan = plan

    def forward(self, todo_list: List[str]) -> str:
        return self._plan.create_todo_list(todo_list)


class _MarkCurrentTodoItemTool(BaseTool):
    """Tool that marks the current (first PENDING/RUNNING) todo item as SUCCESS."""

    def __init__(self, plan: "InMemoryPlan"):
        super().__init__(
            name="mark_current_todo_item",
            description="Mark the current todo item (first PENDING or RUNNING) as completed (SUCCESS).",
        )
        self._plan = plan

    def forward(self) -> str:
        return self._plan.mark_current_todo_item()


class _InsertTodoListTool(BaseTool):
    """Tool that inserts new todo items at a given index."""

    def __init__(self, plan: "InMemoryPlan"):
        super().__init__(
            name="insert_todo_list",
            description="Insert new todo items. Pass a list of task strings and optional index (default -1 = append).",
        )
        self._plan = plan

    def forward(self, todo_list: List[str], index: int = -1) -> str:
        return self._plan.insert_todo_list(todo_list, index)


class _RemoveTodoItemsTool(BaseTool):
    """Tool that removes todo items by indices."""

    def __init__(self, plan: "InMemoryPlan"):
        super().__init__(
            name="remove_todo_items",
            description="Remove todo items by their indices (0-based). Pass a list of indices to remove.",
        )
        self._plan = plan

    def forward(self, indices: List[int]) -> str:
        return self._plan.remove_todo_items(indices)


class InMemoryPlan(PygentOperator):
    """In-memory plan: a todo list with create/mark/insert/remove tools. List stored as list of dicts for serialization."""

    todo_list: PygentList

    def __init__(self, **kwargs):
        super().__init__()
        # Ensure todo_list is a list (of dicts: {"content": str, "status": int})
        if not self.todo_list:
            self.todo_list.data = []

    def get_tools(self) -> List[BaseTool]:
        return [
            _CreateTodoListTool(self),
            _MarkCurrentTodoItemTool(self),
            _InsertTodoListTool(self),
            _RemoveTodoItemsTool(self),
        ]

    def create_todo_list(self, todo_list: List[str]) -> str:
        self.todo_list.clear()
        for s in todo_list:
            self.todo_list.append({"content": s, "status": PygentStatus.PENDING})
        return f"Created todo list with {len(todo_list)} item(s)."

    def mark_current_todo_item(self) -> str:
        for i, item in enumerate(self.todo_list):
            d = item if isinstance(item, dict) else item.to_dict()
            if d.get("status") in (PygentStatus.PENDING, PygentStatus.RUNNING):
                d["status"] = PygentStatus.SUCCESS
                self.todo_list[i] = d
                return f"Marked item {i} as completed."
        return "No PENDING or RUNNING item to mark."

    def insert_todo_list(self, todo_list: List[str], index: int = -1) -> str:
        new_items = [{"content": s, "status": PygentStatus.PENDING} for s in todo_list]
        if index == -1:
            self.todo_list.extend(new_items)
        else:
            for j, item in enumerate(new_items):
                self.todo_list.insert(index + j, item)
        return f"Inserted {len(todo_list)} item(s)."

    def remove_todo_items(self, indices: List[int]) -> str:
        if not indices:
            return "No indices given."
        # Remove from highest index first to avoid shifting
        removed = 0
        for i in sorted(set(indices), reverse=True):
            if 0 <= i < len(self.todo_list):
                self.todo_list.pop(i)
                removed += 1
        return f"Removed {removed} item(s)."
