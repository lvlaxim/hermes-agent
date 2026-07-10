"""Agent internals -- extracted modules from run_agent.py.

These modules contain pure utility functions and self-contained classes
that were previously embedded in the 3,600-line run_agent.py. Extracting
them makes run_agent.py focused on the AIAgent orchestrator class.
"""

from . import jiter_preload as _jiter_preload  # noqa: F401
from . import provider_response_diagnostics as _provider_response_diagnostics  # noqa: F401
