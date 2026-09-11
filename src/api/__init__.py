"""
FastAPI Application package.
Note: do not bind `app` on this package in a way that breaks `import src.api.app`.
"""

from . import app as app_module

application = app_module.app

__all__ = ["application", "app_module"]
