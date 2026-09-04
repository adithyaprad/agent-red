"""Finding the data files this repository carries, from wherever the package was installed.

Three things live beside the code rather than inside it: the hand-authored shop in
`data/store/`, the technique corpus in `data/techniques/`, and the registry of targets this
installation is permitted to attack. All three are content somebody reads and edits, so they
stay at the top of the repository where they can be found, rather than being buried in the
package directory.

That costs one thing, and it is the thing this module exists to pay. `Path(__file__)` locates
the package, and the package is not always inside the working tree: installed into a container's
site-packages, walking up from the module lands in `/usr/local/lib/python3.12` and every one of
those paths resolves to nothing. The failure is late and it reads like corruption rather than
like a layout problem, because the first thing to notice is a missing `catalog.json` several
frames into starting a server.

So a location is looked for rather than computed. The working tree the package was imported from
is tried first, then the working directory, and an environment variable overrides both for the
case neither describes. Nothing is created and nothing is guessed at: a path that resolves
nowhere raises with every location that was tried, because a tool server that starts against a
shop it could not find is worse than one that refuses to start.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV_VAR = "AGENTRED_ROOT"
"""Where the repository's data files are, when neither default describes it."""

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
"""The working tree, when the package is imported from `src/` inside one.

`src/agentred/paths.py` sits three levels below the repository root. Installed into
site-packages this points at the environment instead, which is why it is a candidate rather
than the answer.
"""


def candidate_roots(env: dict[str, str] | None = None) -> tuple[Path, ...]:
    """Every place the repository's data files might be, in the order they are tried.

    Args:
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        The override if one is set, then the working tree the package came from, then the working
        directory. Ordered so an explicit answer wins and a container's `WORKDIR` still works.
    """
    env = os.environ if env is None else env
    override = env.get(ROOT_ENV_VAR, "").strip()
    roots = [Path(override)] if override else []
    roots.append(PACKAGE_ROOT)
    roots.append(Path.cwd())
    return tuple(roots)


def repo_path(*parts: str, env: dict[str, str] | None = None) -> Path:
    """Locate one of the repository's data files or directories.

    Args:
        *parts: Path segments below the repository root, for example `"data"`, `"store"`.
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        The first candidate that exists.

    Raises:
        FileNotFoundError: If no candidate exists, naming every location tried and the
            variable that overrides them. A caller cannot recover from this and should not
            try: the alternative to a clear error here is a server serving an empty shop, in
            which every declared rule reports as never in play and the run reads as clean.
    """
    tried = [root.joinpath(*parts) for root in candidate_roots(env)]
    for path in tried:
        if path.exists():
            return path
    locations = ", ".join(str(path) for path in tried)
    raise FileNotFoundError(
        f"{'/'.join(parts)} is not in any of: {locations}. "
        f"Set {ROOT_ENV_VAR} to the working tree holding it."
    )
