"""Streamlit UI package for the deep-research orchestrator.

Submodules are imported explicitly by callers (e.g. ``from ui.semantic_schema
import load_semantic_schema``) — this package init intentionally does no
eager imports so that pure-logic modules can be loaded without dragging the
orchestrator runtime in.
"""
