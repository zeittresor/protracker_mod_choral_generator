from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTextBrowser,
    QToolBar,
)


_NODE_RE = re.compile(r'^\s*@NODE\s+(\S+)\s+"([^"]*)"\s*$', re.IGNORECASE)
_DB_RE = re.compile(r'^\s*@DATABASE\s+"([^"]*)"\s*$', re.IGNORECASE)


@dataclass
class GuideNode:
    node_id: str
    title: str
    raw_lines: List[str]


class GuideDoc:
    """
    Minimal AmigaGuide parser sufficient for our docs/*.guide.
    Supports:
      - @DATABASE "Title"
      - @WORDWRAP
      - @NODE ID "Title" ... @ENDNODE
      - inline macros:
          @{b} / @{ub}
          @{"Text" LINK TARGET}
    """

    def __init__(self, path: Path):
        self.path = path
        self.database_title: str = path.name
        self.nodes: Dict[str, GuideNode] = {}
        self.wordwrap: bool = True
        self._parse()

    def _parse(self) -> None:
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        cur_id: Optional[str] = None
        cur_title: str = ""
        cur_buf: List[str] = []

        for ln in lines:
            mdb = _DB_RE.match(ln)
            if mdb:
                self.database_title = mdb.group(1)
                continue

            if ln.strip().upper() == "@WORDWRAP":
                self.wordwrap = True
                continue

            mnode = _NODE_RE.match(ln)
            if mnode:
                # flush any open node (defensive)
                if cur_id is not None:
                    self.nodes[cur_id] = GuideNode(cur_id, cur_title, cur_buf)
                cur_id = mnode.group(1)
                cur_title = mnode.group(2)
                cur_buf = []
                continue

            if ln.strip().upper() == "@ENDNODE":
                if cur_id is not None:
                    self.nodes[cur_id] = GuideNode(cur_id, cur_title, cur_buf)
                cur_id = None
                cur_title = ""
                cur_buf = []
                continue

            if ln.strip().upper().startswith("@ENDDATABASE"):
                break

            if cur_id is not None:
                cur_buf.append(ln)

        # flush tail
        if cur_id is not None:
            self.nodes[cur_id] = GuideNode(cur_id, cur_title, cur_buf)

    def default_node(self) -> str:
        if "MAIN" in self.nodes:
            return "MAIN"
        if self.nodes:
            return next(iter(self.nodes.keys()))
        return "MAIN"



class _GuideBrowser(QTextBrowser):
    """Clickable QTextBrowser that reliably routes link clicks to our handler.

    Some Qt/PyQt builds don't emit anchorClicked for custom schemes consistently.
    We therefore also detect anchors on mouse release.
    """

    def __init__(self, on_click):
        super().__init__()
        self._on_click = on_click
        self.setOpenExternalLinks(False)
        try:
            self.setOpenLinks(False)
        except Exception:
            pass
        # ensure link interaction is enabled
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.setReadOnly(True)

    def mouseReleaseEvent(self, e):
        try:
            btn = e.button()
        except Exception:
            btn = None
        if btn == Qt.MouseButton.LeftButton:
            try:
                pos = e.position().toPoint()
            except Exception:
                try:
                    pos = e.pos()
                except Exception:
                    pos = None
            if pos is not None:
                href = self.anchorAt(pos)
                if href:
                    # Route through the same handler as anchorClicked
                    if self._on_click is not None:
                        self._on_click(QUrl(href))
                    e.accept()
                    return
        super().mouseReleaseEvent(e)


class GuideViewer(QMainWindow):
    def __init__(self, start_file: Path, start_node: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("AmigaGuide")
        self.setMinimumSize(840, 640)

        self._cache: Dict[Path, GuideDoc] = {}
        self._history: List[Tuple[Path, str]] = []
        self._hist_idx: int = -1

        self._current_file: Optional[Path] = None
        self._current_node: Optional[str] = None

        # UI
        self.title_label = QLabel("")
        self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.browser = _GuideBrowser(self._on_anchor_clicked)
        self.browser.anchorClicked.connect(self._on_anchor_clicked)

        # Ensure keyboard shortcuts (Zoom etc.) hit the document widget and that
        # it can actually receive key events.
        self.browser.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.browser.setFocus()

        self.setCentralWidget(self.browser)
        self._build_menus()
        self._build_toolbar()

        sb = QStatusBar()
        self.setStatusBar(sb)

        # Workbench-ish styling (light bevel, mono text)
        self._apply_workbench_style()

        # Load start
        node = start_node
        self.navigate_to(start_file, node, push=True)

    def _apply_workbench_style(self) -> None:
        # Simple "Workbench-ish" look; user can still theme main app independently.
        self.setStyleSheet(
            """
            QMainWindow { background: #b0b0b0; }
            QToolBar { background: #b0b0b0; border-bottom: 1px solid #707070; }
            QStatusBar { background: #b0b0b0; border-top: 1px solid #707070; }
            QTextBrowser {
                background: #c8c8c8;
                border: 2px solid #707070;
                border-top-color: #ffffff;
                border-left-color: #ffffff;
                padding: 10px;
                font-family: "Courier New", monospace;
                font-size: 12pt;
            }
            a { color: #0000aa; text-decoration: none; }
            a:hover { text-decoration: underline; }
            """
        )

    def _build_toolbar(self) -> None:
        tb = QToolBar("Navigation")
        tb.setMovable(False)

        self.act_back = QAction("Back", self)
        self.act_back.setShortcut(QKeySequence.StandardKey.Back)
        self.act_back.triggered.connect(self.go_back)

        self.act_fwd = QAction("Forward", self)
        self.act_fwd.setShortcut(QKeySequence.StandardKey.Forward)
        self.act_fwd.triggered.connect(self.go_forward)

        self.act_home = QAction("Home", self)
        self.act_home.setShortcut(QKeySequence("Alt+Home"))
        self.act_home.triggered.connect(self.go_home)

        tb.addAction(self.act_back)
        tb.addAction(self.act_fwd)
        tb.addSeparator()
        tb.addAction(self.act_home)
        tb.addSeparator()
        tb.addWidget(self.title_label)

        self.addToolBar(tb)
        self._sync_nav_actions()

    def _build_menus(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("&File")
        act_open = QAction("&Open…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_dialog)
        m_file.addAction(act_open)

        act_open_index = QAction("Open &Index", self)
        act_open_index.triggered.connect(self._open_index)
        m_file.addAction(act_open_index)

        m_file.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        m_nav = mb.addMenu("&Navigate")
        act_back = QAction("&Back", self); act_back.setShortcut(QKeySequence.StandardKey.Back); act_back.triggered.connect(self.go_back)
        act_fwd = QAction("&Forward", self); act_fwd.setShortcut(QKeySequence.StandardKey.Forward); act_fwd.triggered.connect(self.go_forward)
        act_home = QAction("&Home", self); act_home.setShortcut(QKeySequence("Alt+Home")); act_home.triggered.connect(self.go_home)
        m_nav.addAction(act_back); m_nav.addAction(act_fwd); m_nav.addSeparator(); m_nav.addAction(act_home)

        m_view = mb.addMenu("&View")
        # Zoom shortcuts vary across keyboard layouts (esp. Ctrl+'+' vs Ctrl+'=').
        act_zoom_in = QAction("Zoom &In", self)
        act_zoom_in.setShortcuts([
            QKeySequence.StandardKey.ZoomIn,
            QKeySequence("Ctrl++"),
            QKeySequence("Ctrl+="),
        ])
        act_zoom_in.triggered.connect(lambda: self.browser.zoomIn(1))

        act_zoom_out = QAction("Zoom &Out", self)
        act_zoom_out.setShortcuts([
            QKeySequence.StandardKey.ZoomOut,
            QKeySequence("Ctrl+-"),
        ])
        act_zoom_out.triggered.connect(lambda: self.browser.zoomOut(1))
        act_zoom_reset = QAction("&Reset Zoom", self); act_zoom_reset.triggered.connect(self._reset_zoom)
        m_view.addAction(act_zoom_in); m_view.addAction(act_zoom_out); m_view.addAction(act_zoom_reset)

        m_help = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._about)
        m_help.addAction(act_about)

    def _reset_zoom(self) -> None:
        # QTextBrowser has no direct reset; approximate by setting a base font size via zoom steps.
        # We'll just recreate the widget zoom to 0 by reloading current content.
        cur = (self._current_file, self._current_node)
        if cur[0] is not None and cur[1] is not None:
            self.navigate_to(cur[0], cur[1], push=False)

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "About AmigaGuide Viewer",
            "Minimal AmigaGuide-like viewer for the Protracker Music Generator manuals.\n"
            "Supports nodes, inline links and basic formatting.\n",
        )

    def _open_dialog(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Open .guide", str(Path.cwd()), "AmigaGuide (*.guide);;All files (*)")
        if fn:
            self.navigate_to(Path(fn), None, push=True)

    def _open_index(self) -> None:
        # try to locate project docs/ index relative to this module
        root = Path(__file__).resolve().parents[2]
        idx = root / "docs" / "ProtrackerMusicGenerator_Index.guide"
        if idx.exists():
            self.navigate_to(idx, None, push=True)
        else:
            QMessageBox.warning(self, "Index not found", f"Could not find:\n{idx}")

    def _get_doc(self, path: Path) -> GuideDoc:
        path = path.resolve()
        if path not in self._cache:
            self._cache[path] = GuideDoc(path)
        return self._cache[path]

    def navigate_to(self, file_path: Path, node_id: Optional[str], push: bool = True) -> None:
        file_path = file_path.resolve()
        if not file_path.exists():
            QMessageBox.warning(self, "Guide not found", f"File not found:\n{file_path}")
            return

        doc = self._get_doc(file_path)
        nid = node_id or doc.default_node()
        if nid not in doc.nodes:
            # robust case-insensitive lookup (Qt URLs may lowercase tokens)
            nid_u = nid.upper()
            if nid_u in doc.nodes:
                nid = nid_u
            else:
                nid_l = nid.lower()
                hit = None
                for k in doc.nodes.keys():
                    if k.lower() == nid_l:
                        hit = k
                        break
                nid = hit or doc.default_node()

        self._current_file = file_path
        self._current_node = nid

        if push:
            # trim forward history
            if self._hist_idx < len(self._history) - 1:
                self._history = self._history[: self._hist_idx + 1]
            self._history.append((file_path, nid))
            self._hist_idx = len(self._history) - 1

        self._render(doc, nid)
        self._sync_nav_actions()

    def _render(self, doc: GuideDoc, node_id: str) -> None:
        node = doc.nodes.get(node_id)
        if not node:
            self.browser.setHtml("<b>Node not found.</b>")
            self.title_label.setText("")
            self.statusBar().showMessage("")
            return

        self.title_label.setText(f"{node.title}")
        self.setWindowTitle(f"AmigaGuide - {doc.database_title}")

        html_body = self._to_html(doc, node)
        self.browser.setHtml(html_body)
        self.statusBar().showMessage(f"{doc.path.name}  ::  {node.node_id}")

    def _to_html(self, doc: GuideDoc, node: GuideNode) -> str:
        css = """
        <style>
        body { font-family: 'Courier New', monospace; font-size: 12pt; white-space: pre-wrap; }
        .sep { color: #404040; }
        </style>
        """
        lines_html: List[str] = []
        for ln in node.raw_lines:
            lines_html.append(self._line_html(ln))
        body = "<br>".join(lines_html)
        return f"<!doctype html><html><head>{css}</head><body>{body}</body></html>"

    _MACRO_RE = re.compile(r'@\{[^}]*\}')

    _LINK_INNER_RE = re.compile(r'^"([^"]+)"\s+LINK\s+(.+)$', re.IGNORECASE)

    def _line_html(self, line: str) -> str:
        parts = []
        last = 0
        for m in self._MACRO_RE.finditer(line):
            if m.start() > last:
                parts.append(html.escape(line[last:m.start()]))
            macro = m.group(0)
            parts.append(self._macro_to_html(macro))
            last = m.end()
        if last < len(line):
            parts.append(html.escape(line[last:]))

        # normalize a few ascii separators for readability
        out = "".join(parts)
        out = out.replace("|", '<span class="sep"> | </span>')
        return out

    def _macro_to_html(self, macro: str) -> str:
        inner = macro[2:-1]  # strip "@{" and "}"
        inner_stripped = inner.strip()

        if inner_stripped.lower() == "b":
            return "<b>"
        if inner_stripped.lower() == "ub":
            return "</b>"

        # link macro?
        m = self._LINK_INNER_RE.match(inner_stripped)
        if m:
            text = m.group(1)
            target = m.group(2).strip()
            # remove surrounding quotes
            if (target.startswith('"') and target.endswith('"')) or (target.startswith("'") and target.endswith("'")):
                target = target[1:-1]
            # IMPORTANT: Do NOT use "ag://..." here.
            # Qt normalizes the host portion of URLs to lowercase. That breaks
            # case-sensitive node ids like "DE"/"EN"/"FR" (they become "de").
            # Using "ag:TARGET" keeps the target in the path portion and avoids
            # host normalization.
            href = f"ag:{target}"
            return f'<a href="{href}">{html.escape(text)}</a>'

        # unknown macro: show as faint text
        return f'<span style="color:#606060;">{html.escape(macro)}</span>'

    def _on_anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() != "ag":
            return

        # Support both old (ag://...) and new (ag:...) link formats.
        s = url.toString()
        if s.startswith("ag://"):
            target = s[5:]
        elif s.startswith("ag:"):
            target = s[3:]
        else:
            target = url.path().lstrip("/")

        self._open_target(target)

    def _open_target(self, target: str) -> None:
        target = target.strip()
        if not target:
            return

        cur_file = self._current_file or Path.cwd()
        base_dir = cur_file.parent

        # external file?
        if ".guide" in target:
            # formats:
            #   file.guide/NODE
            #   file.guide
            if "/" in target:
                file_part, node_part = target.split("/", 1)
            else:
                file_part, node_part = target, None

            p = Path(file_part)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            self.navigate_to(p, node_part, push=True)
            return

        # node within current file
        if self._current_file is not None:
            self.navigate_to(self._current_file, target, push=True)

    def _sync_nav_actions(self) -> None:
        can_back = self._hist_idx > 0
        can_fwd = 0 <= self._hist_idx < len(self._history) - 1
        self.act_back.setEnabled(can_back)
        self.act_fwd.setEnabled(can_fwd)
        self.act_home.setEnabled(True)

    def go_back(self) -> None:
        if self._hist_idx <= 0:
            return
        self._hist_idx -= 1
        f, n = self._history[self._hist_idx]
        self.navigate_to(f, n, push=False)

    def go_forward(self) -> None:
        if self._hist_idx >= len(self._history) - 1:
            return
        self._hist_idx += 1
        f, n = self._history[self._hist_idx]
        self.navigate_to(f, n, push=False)

    def go_home(self) -> None:
        if not self._history:
            return
        f, _ = self._history[0]
        self.navigate_to(f, "MAIN", push=True)
