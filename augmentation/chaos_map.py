"""Backwards-compatibility shim.

The augmentation module was renamed from ``chaos_map`` to ``chaos_augment``,
but several call sites still import ``augmentation.chaos_map`` (demo_offline.py,
main.py, rag_fca/knowledge_base.py). This module re-exports everything from
``chaos_augment`` so those imports keep working and the code remains runnable.

New code should import from ``augmentation.chaos_augment`` directly.
"""

from augmentation.chaos_augment import *  # noqa: F401,F403
from augmentation import chaos_augment as _impl

# Re-export module-level names that ``import *`` may skip (those without
# an ``__all__`` entry or starting with an underscore are intentionally omitted).
__all__ = [name for name in dir(_impl) if not name.startswith("_")]
