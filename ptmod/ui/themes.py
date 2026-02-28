from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtWidgets import QApplication

@dataclass(frozen=True)
class Theme:
    name: str
    qss: str
    font_family: Optional[str] = None
    font_size: Optional[int] = None

def _qss_protracker_gray() -> str:
    # Close to the old Tk 'ProTracker vibe' (gray + Courier, beveled borders)
    return r'''
    QWidget { background: #8f8f8f; color: #000000; font-weight: bold; }
    QTabWidget::pane { border: 2px solid #000000; }
    QTabBar::tab {
        background: #7f7f7f;
        border: 2px solid #000000;
        padding: 6px 12px;
        margin-right: 2px;
    }
    QTabBar::tab:selected { background: #9b9b9b; }
    QGroupBox {
        border: 2px solid #000000;
        margin-top: 10px;
        padding: 10px;
        background: #8f8f8f;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
    QPushButton {
        background: #9b9b9b;
        border: 2px solid #000000;
        padding: 6px 10px;
    }
    QPushButton:hover { background: #b0b0b0; }
    QPushButton:pressed { background: #7f7f7f; }
    QPushButton:disabled { background: #7f7f7f; color: #555555; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
        background: #9b9b9b;
        border: 2px solid #000000;
        selection-background-color: #0000aa;
        padding: 4px;
    }
    QCheckBox { spacing: 6px; }
    QProgressBar { border: 2px solid #000000; text-align: center; background: #9b9b9b; }
    QProgressBar::chunk { background: #00aa00; }
    '''

def _qss_ecs() -> str:
    # Chunky, high-contrast, classic Workbench-ish vibe
    return r'''
    QWidget { background: #003399; color: #FFFFFF; }
    QTabWidget::pane { border: 2px solid #FFFFFF; }
    QTabBar::tab {
        background: #001F66; color: #FFFFFF;
        border: 2px solid #FFFFFF;
        padding: 6px 12px;
        margin-right: 2px;
    }
    QTabBar::tab:selected { background: #0055CC; }
    QGroupBox {
        border: 2px solid #FFFFFF;
        margin-top: 10px;
        padding: 8px;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
    QPushButton {
        background: #0055CC;
        border: 2px solid #FFFFFF;
        padding: 6px 10px;
    }
    QPushButton:pressed { background: #0077FF; }
    QPushButton:disabled { background: #001F66; color: #99B3FF; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
        background: #001F66;
        border: 2px solid #FFFFFF;
        selection-background-color: #0077FF;
        padding: 4px;
    }
    QProgressBar { border: 2px solid #FFFFFF; text-align: center; }
    QProgressBar::chunk { background: #00DD00; }
    '''

def _qss_mui() -> str:
    # Soft gray, beveled controls (approximate MUI look)
    return r'''
    QWidget { background: #C8C8C8; color: #111111; }
    QTabWidget::pane { border: 1px solid #666666; }
    QTabBar::tab {
        background: #B0B0B0;
        border: 1px solid #666666;
        padding: 6px 12px;
        margin-right: 1px;
    }
    QTabBar::tab:selected { background: #E0E0E0; }
    QGroupBox {
        border: 1px solid #777777;
        border-radius: 4px;
        margin-top: 10px;
        padding: 8px;
        background: #D6D6D6;
    }
    QPushButton {
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #EEEEEE, stop:1 #B8B8B8);
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px 10px;
    }
    QPushButton:pressed {
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #B8B8B8, stop:1 #EEEEEE);
    }
    QPushButton:disabled { background: #B0B0B0; color: #777777; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
        background: #F4F4F4;
        border: 1px solid #666666;
        border-radius: 3px;
        selection-background-color: #99CCFF;
        padding: 4px;
    }
    QProgressBar { border: 1px solid #666666; text-align: center; background: #F4F4F4; }
    QProgressBar::chunk { background: #4CAF50; }
    '''

def _qss_modern_dark() -> str:
    return r'''
    QWidget { background: #1E1E1E; color: #E6E6E6; }
    QTabWidget::pane { border: 1px solid #3A3A3A; }
    QTabBar::tab { background: #2A2A2A; border: 1px solid #3A3A3A; padding: 7px 12px; margin-right: 2px; }
    QTabBar::tab:selected { background: #3A3A3A; }
    QGroupBox { border: 1px solid #3A3A3A; border-radius: 6px; margin-top: 10px; padding: 10px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
    QPushButton { background: #2F6FED; border: 0px; border-radius: 6px; padding: 7px 12px; }
    QPushButton:hover { background: #3A7BFF; }
    QPushButton:pressed { background: #2459C7; }
    QPushButton:disabled { background: #2A2A2A; color: #777777; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background: #2A2A2A; border: 1px solid #3A3A3A; border-radius: 6px; padding: 6px; }
    QProgressBar { border: 1px solid #3A3A3A; border-radius: 6px; text-align: center; background: #2A2A2A; }
    QProgressBar::chunk { border-radius: 6px; background: #2F6FED; }
    '''

def _qss_modern_light() -> str:
    return r'''
    QWidget { background: #F6F7F9; color: #1A1A1A; }
    QTabWidget::pane { border: 1px solid #D6D9DE; }
    QTabBar::tab { background: #EDEFF2; border: 1px solid #D6D9DE; padding: 7px 12px; margin-right: 2px; }
    QTabBar::tab:selected { background: #FFFFFF; }
    QGroupBox { border: 1px solid #D6D9DE; border-radius: 6px; margin-top: 10px; padding: 10px; background: #FFFFFF; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
    QPushButton { background: #2F6FED; color: #FFFFFF; border: 0px; border-radius: 6px; padding: 7px 12px; }
    QPushButton:hover { background: #3A7BFF; }
    QPushButton:pressed { background: #2459C7; }
    QPushButton:disabled { background: #E0E3E8; color: #7A7A7A; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background: #FFFFFF; border: 1px solid #D6D9DE; border-radius: 6px; padding: 6px; }
    QProgressBar { border: 1px solid #D6D9DE; border-radius: 6px; text-align: center; background: #FFFFFF; }
    QProgressBar::chunk { border-radius: 6px; background: #2F6FED; }
    '''

BUILTIN_THEMES: Dict[str, Theme] = {
    "ProTracker Gray": Theme("ProTracker Gray", _qss_protracker_gray(), font_family="Courier New", font_size=10),
    "Modern Dark": Theme("Modern Dark", _qss_modern_dark(), font_family=None, font_size=None),
    "Modern Light": Theme("Modern Light", _qss_modern_light(), font_family=None, font_size=None),
    "Amiga ECS": Theme("Amiga ECS", _qss_ecs(), font_family="Courier New", font_size=10),
    "Amiga MUI": Theme("Amiga MUI", _qss_mui(), font_family="Tahoma", font_size=10),
}

def load_external_themes(themes_dir: Path) -> Dict[str, Theme]:
    themes: Dict[str, Theme] = {}
    if not themes_dir.exists():
        return themes
    for p in themes_dir.glob("*.qss"):
        try:
            themes[p.stem] = Theme(p.stem, p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    return themes

def apply_theme(app: QApplication, theme: Theme):
    app.setStyleSheet(theme.qss)
    if theme.font_family or theme.font_size:
        font = app.font()
        if theme.font_family:
            font.setFamily(theme.font_family)
        if theme.font_size:
            font.setPointSize(int(theme.font_size))
        app.setFont(font)
