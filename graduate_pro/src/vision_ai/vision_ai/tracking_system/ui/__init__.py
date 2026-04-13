# tracking_system/ui/__init__.py
"""
Tracking System UI Module

简化的UI模块，提供图形界面替代命令行交互
"""

from .tracking_ui import TrackingUI, UIState

__version__ = "1.0.0"
__author__ = "Enhanced Tracking System"

__all__ = ["TrackingUI", "UIState"]