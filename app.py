from __future__ import annotations
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from ptmod.ui.themes import BUILTIN_THEMES, load_external_themes, apply_theme
from ptmod.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    # Themes: built-in + themes/*.qss
    themes_dir = Path(__file__).parent / "themes"
    themes = dict(BUILTIN_THEMES)
    themes.update(load_external_themes(themes_dir))

    def _apply(theme):
        apply_theme(app, theme)

    # Default theme
    if "ProTracker Gray" in themes:
        _apply(themes["ProTracker Gray"])
    elif "Modern Dark" in themes:
        _apply(themes["Modern Dark"])
    else:
        _apply(next(iter(themes.values())))

    win = MainWindow(themes=themes, apply_theme_cb=_apply)
    win.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
