# Config (kernel)

Kernel-side configuration surface. It owns only pure shape validation of the
raw configuration mapping (`validate_config_mapping`); the typed settings live
in `lara.schema` (`RunConfig` and friends).

The configuration *mechanism* — reading `~/.lara/config.yaml`, merging
defaults, parsing the YAML subset, and editing files — lives in
`lara_shell.config`. The kernel never imports that module; the shell loader
calls `lara.config.validate_config_mapping` before constructing `RunConfig`.
