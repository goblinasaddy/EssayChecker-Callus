"""Vercel Serverless Function entrypoint for EssayChecker FastAPI Application."""
import os
import sys

# Ensure repository root is on sys.path so that 'src' is directly importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.api.server import app
