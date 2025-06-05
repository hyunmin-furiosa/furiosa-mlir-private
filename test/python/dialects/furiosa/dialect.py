# RUN: %PYTHON

from furiosa_mlir.ir import *
import furiosa_mlir.dialects.furiosa as furiosa
from furiosa_mlir.passmanager import *

with Context(), Location.unknown():
    PassManager.parse("any(furiosa-deallocation)")
