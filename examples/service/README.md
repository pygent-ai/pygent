# Executionnable Agent Service

This application-owned example composes the Pygent 0.2 SDK and executes real managed `invoke()` and `stream()` flows with an offline deterministic model boundary.

- `models.py` owns model routes, retry policy, generation config, and tool projection.
- `tools.py` owns portable tool declarations, application authorization, and executor wiring.
- `agents.py` composes ReAct and review Modules. ReAct commits the draft; the Coordinator appends the reviewed answer without deleting it.
- `app.py` owns Runtime binding, finite deadlines, invoke/stream mapping, and commit timing.
- `domain.py` owns request/response values and a revision-based CAS conversation Store.

```text
CoordinatorAgent
├── ReActLayer
│   ├── ModelCallLayer
│   └── ToolCallLayer
│       └── WeatherAuthorization
└── ReviewAgent
    └── ModelCallLayer
```

Execution it from the repository root:

```bash
uv run python -m examples.service.main
uv run pytest -q tests/integration/test_service_execution.py
```

The example demonstrates that direct business results remain `(message, context)`, stream events are observation-only, and the application—not Runtime—owns the final CAS commit.
