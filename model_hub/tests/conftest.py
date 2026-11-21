"""Pytest configuration for model_hub tests."""
import os
import sys

# Ensure model_hub is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
