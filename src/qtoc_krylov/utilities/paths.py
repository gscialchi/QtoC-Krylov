from pathlib import Path


def _find_root():
    here = Path(__file__).resolve()
    for path in [here, *here.parents]:
        if (path / "pyproject.toml").exists():
            return str(path)
    raise RuntimeError("Project root not found.")


ROOT_DIR = _find_root()
STORE_DIR = ROOT_DIR + "/data/operators/"
DOER_DIR = ROOT_DIR + "/data/calculations/"
FIG_DIR = ROOT_DIR + '/figures/'
CONFIG_DIR = ROOT_DIR + '/configs/'
