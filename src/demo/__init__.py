from demo.utils import find_project_root

from .ant import main as ant_demo
from .reacher import main as reacher_demo

ROOT = find_project_root()

CHEKPOINT_PATH = ROOT / "data"


def main() -> None:
    reacher_demo(str(CHEKPOINT_PATH / "latest.onnx"))
