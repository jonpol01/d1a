### Added

- Training recipes (#63): `python -m d1a.recipe run <recipe.yaml> --out <dir> --device <device>` runs a versioned YAML
  file of stages (`format: d1a-recipe`, version 1), each one `d1a.train` run with the options it lists, a later stage
  starting from the previous stage's checkpoint. Every option is checked by `d1a.train`'s own parser before anything
  runs, machine settings (device, resume, save interval) come from the command, and each stage's directory gets
  `recipe.json` (the recipe, its sha256, the stage and the exact command). `--dry-run` prints the commands.
  `recipes/d1a-e2b.yaml` is the README's E2B run. `d1a.train`'s `parse_args` takes an argument list; PyYAML is now a
  declared dependency.
