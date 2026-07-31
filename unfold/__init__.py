"""UnfoldAll package root."""

try:
    import gymnasium as gym

    gym.register(
        id="UnfoldAll-Cloth-Direct-v0",
        entry_point="unfold.simulation.env:Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "configs:config.yaml",
        },
    )
except ModuleNotFoundError:
    pass
