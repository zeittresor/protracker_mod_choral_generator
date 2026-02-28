from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from ptmod.ui.guide_viewer import GuideViewer


def main() -> int:
    if len(sys.argv) >= 2:
        start = Path(sys.argv[1])
    else:
        start = Path(__file__).parent / "docs" / "ProtrackerMusicGenerator_Index.guide"

    start = start.resolve()
    app = QApplication(sys.argv)
    win = GuideViewer(start_file=start, start_node=None)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
