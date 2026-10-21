"""The exact number the helper library computes with."""

from fractions import Fraction


def exact(x: float) -> Fraction:
    """The shortest decimal that reads back as ``x``, exactly: the number RFC 8785 writes."""
    return Fraction(repr(x))
