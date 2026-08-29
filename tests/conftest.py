"""Test-wide setup.

`agentred.targets.runtime` asserts test mode at import, which is the point of it. Setting
the variable here rather than in each test keeps that assertion real: remove this line and
every target test fails, which is the behaviour the assertion promises.
"""

import os

os.environ.setdefault("AGENTRED_TARGET_MODE", "test")
