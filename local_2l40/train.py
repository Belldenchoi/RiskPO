"""Original RiskPO, notebook-matched research settings, two NVIDIA L40 GPUs."""
from pathlib import Path
import sys

SHARED = Path(__file__).resolve().parents[1] / 'local_4a100'
# Use a named module so importing this wrapper cannot shadow the shared train.py.
import importlib.util
sys.path.insert(0, str(SHARED))
spec = importlib.util.spec_from_file_location('riskpo_shared_runner', SHARED / 'train.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


if __name__ == '__main__':
    # Explicit flags later in argv take precedence, e.g. --hardware-profile l40_2_safe.
    runner.main(['--hardware-profile', 'l40_2_fast', *sys.argv[1:]])
