"""Private signal distinguishing framework budgets from business timeouts."""


class _ExecutionDeadlineExpired(TimeoutError):
    """Only execution and policy budget boundaries may create this signal."""
