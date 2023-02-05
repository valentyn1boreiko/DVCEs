from math import pi

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy
except ImportError:
    numpy = None

if numpy is None and torch is None:
    raise ImportError("Must have either Numpy or PyTorch but both not found")


def set_framework_dependencies(x):
    if type(x) is numpy.ndarray:
        def to_dtype(a):
            return a
        fw = numpy
    else:
        def to_dtype(a):
            return a.to(x.dtype)
        fw = torch
    eps = fw.finfo(fw.float32).eps
    return fw, to_dtype, eps


def cubic(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    absx = fw.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1.0) * to_dtype(absx <= 1.0) + (
        -0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0
    ) * to_dtype((1.0 < absx) & (absx <= 2.0))


def lanczos2(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (
        (fw.sin(pi * x) * fw.sin(pi * x / 2) + eps) / ((pi**2 * x**2 / 2) + eps)
    ) * to_dtype(abs(x) < 2)


def lanczos3(x, fw):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (
        (fw.sin(pi * x) * fw.sin(pi * x / 3) + eps) / ((pi**2 * x**2 / 3) + eps)
    ) * to_dtype(abs(x) < 3)


def linear(x, fw):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (x + 1) * to_dtype((-1 <= x) & (x < 0)) + (1 - x) * to_dtype(
        (0 <= x) & (x <= 1)
    )


def box(x, fw):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return to_dtype((-1 <= x) & (x < 0)) + to_dtype((0 <= x) & (x <= 1))
