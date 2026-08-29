"""The agents under test.

A declarative spec becomes a Claude Agent SDK agent behind an HTTP endpoint. This package
stands in for a merchant agent platform: it implements the contract in `spec/`, it never
owns it, and nothing in `judge/` may import from here.
"""
