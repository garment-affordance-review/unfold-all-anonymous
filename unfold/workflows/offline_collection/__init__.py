"""Offline collection pipeline package.

Keep package import side-effect free.

Command entrypoints should be imported from their modules directly, instead of
eagerly importing collection scripts here.
"""

__all__: list[str] = []
