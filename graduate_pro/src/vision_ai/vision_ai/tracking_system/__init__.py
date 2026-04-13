# tracking/__init__.py
"""
Vision AI Tracking Module
智能追踪模块 - 支持自适应学习和在线权重调整
"""

from .tracking_node import TrackingNode
from .enhanced_tracker import EnhancedTracker

__version__ = "1.0.0"
__all__ = ["TrackingNode", "EnhancedTracker"]