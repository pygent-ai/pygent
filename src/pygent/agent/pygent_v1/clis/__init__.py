"""仓内挂载（vendored）CLI 工具及其到用户 CLIs 目录的自动 provision。

本包不是运行时子系统，而是随包分发的 CLI 工具的宿主：
``dynamic-memory-cli/dynamic_memory_cli.py`` 是永续会话记忆管道的单文件
CLI，首次使用时由 :func:`pygent_v1.clis.provision.ensure_dynamic_memory_cli`
复制到 ``<user pygent root>/clis/dynamic-memory-cli/``。CLIs 目录不参与
agent 技能目录扫描；各角色的命令面由代码内的指令信封注入。
"""
