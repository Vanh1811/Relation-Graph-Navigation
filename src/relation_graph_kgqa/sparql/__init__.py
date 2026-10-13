"""Lightweight SPARQL structural parsing."""

from .parser import QueryFeatures, TriplePattern, parse_query

__all__ = ["QueryFeatures", "TriplePattern", "parse_query"]
