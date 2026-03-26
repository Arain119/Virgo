"""Policy networks and agent wiring for Virgo Trader.

This package intentionally keeps imports lightweight at module import time. Heavy
ML dependencies (e.g., torch) should be imported by the specific submodules that
need them.
"""
