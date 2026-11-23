"""Reading of the archived PPO checkpoints."""
from __future__ import annotations

import pickle
from pathlib import Path

_TRUSTED = {
    "numpy_agents": {"PPOAgent", "MLP", "Adam"},
    "numpy.random._pickle": {"__generator_ctor", "__bit_generator_ctor"},
    "numpy.random._pcg64": {"PCG64"},
    "numpy": {"ndarray", "dtype"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.numeric": {"_frombuffer"},
    "numpy.random.bit_generator": {"SeedSequence", "__pyx_unpickle_SeedSequence"},
}


class _CheckpointReader(pickle.Unpickler):
    def find_class(self, module, name):
        if name not in _TRUSTED.get(module, set()):
            raise ValueError(f"Unexpected serialized object {module}.{name}")
        return super().find_class(module, name)


def load_checkpoint(path):
    """Return the PPO policies of a checkpoint: a list of three or one joint policy."""
    with Path(path).open("rb") as stream:
        return _CheckpointReader(stream).load()
