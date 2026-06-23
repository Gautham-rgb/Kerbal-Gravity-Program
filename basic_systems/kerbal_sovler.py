import numpy as np
import numba as nb
from typing import Optional
import math
pi = math.pi

body_type = nb.deferred_type()

body_spec = [
    ("mu", nb.float64),
    ("radius", nb.float64),
    ("semi_major_axis", nb.float64),
    ("eccen", nb.float64),
    ("phase_offset", nb.float64),
    ("parent", nb.optional(body_type))
]

@nb.experimental.jitclass(spec = body_spec)
class Body:
    def __init__(self, mu, radius, semi_major_axis = 0.0, eccen = 0.0, phase_offset = 0.0, parent = None):
        self.mu = mu
        self.radius = radius
        self.semi_major_axis = semi_major_axis
        self.eccen = eccen
        self.parent = parent
        self.phase_offset = phase_offset

        if not self.parent:
            self.semi_major_axis = 0.0
            self.eccen = 0.0
            self.phase_offset = 0.0