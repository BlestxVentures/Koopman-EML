"""
koopman_eml -- Koopman operator lifting via EML observable dictionaries.

The EML (Exp-Minus-Log) operator eml(x, y) = exp(x) - ln(y), paired with
the constant 1, generates every standard elementary function (Odrzywołek, 2026).
This package uses parameterized EML trees as a gradient-trainable, symbolically-
interpretable dictionary for Koopman operator approximation.
"""

from koopman_eml.eml_ops import eml, eml_numpy, taylor_exp, taylor_ln
from koopman_eml.eml_tree import EMLNode, EMLTree, EMLTreeVectorized
from koopman_eml.koopman_model import KoopmanEML

__version__ = "0.1.0"

__all__ = [
    "eml",
    "eml_numpy",
    "taylor_exp",
    "taylor_ln",
    "EMLNode",
    "EMLTree",
    "EMLTreeVectorized",
    "KoopmanEML",
]
