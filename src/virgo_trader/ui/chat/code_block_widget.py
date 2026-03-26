"""Code block rendering helpers for the chat UI.

Adds syntax highlighting and copy-to-clipboard support for assistant responses.
"""

from PyQt6.QtCore import QRegularExpression, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """A syntax highlighter for Python code."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#ff79c6"))
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            "\\bFalse\\b",
            "\\bNone\\b",
            "\\bTrue\\b",
            "\\band\\b",
            "\\bas\\b",
            "\\bassert\\b",
            "\\basync\\b",
            "\\bawait\\b",
            "\\bbreak\\b",
            "\\bclass\\b",
            "\\bcontinue\\b",
            "\\bdef\\b",
            "\\bdel\\b",
            "\\belif\\b",
            "\\belse\\b",
            "\\bexcept\\b",
            "\\bfinally\\b",
            "\\bfor\\b",
            "\\bfrom\\b",
            "\\bglobal\\b",
            "\\bif\\b",
            "\\bimport\\b",
            "\\bin\\b",
            "\\bis\\b",
            "\\blambda\\b",
            "\\bnonlocal\\b",
            "\\bnot\\b",
            "\\bor\\b",
            "\\bpass\\b",
            "\\braise\\b",
            "\\breturn\\b",
            "\\btry\\b",
            "\\bwhile\\b",
            "\\bwith\\b",
            "\\byield\\b",
            "\\bself\\b",
        ]
        self.highlighting_rules.extend(
            [(QRegularExpression(pattern), keyword_format) for pattern in keywords]
        )

        # Decorators
        decorator_format = QTextCharFormat()
        decorator_format.setForeground(QColor("#bd93f9"))
        self.highlighting_rules.append((QRegularExpression("@\\w+"), decorator_format))

        # Built-in functions and types
        builtin_format = QTextCharFormat()
        builtin_format.setForeground(QColor("#8be9fd"))
        builtin_format.setFontItalic(True)
        builtins = [
            "\\babs\\b",
            "\\ball\\b",
            "\\bany\\b",
            "\\bascii\\b",
            "\\bbin\\b",
            "\\bbool\\b",
            "\\bbytearray\\b",
            "\\bbytes\\b",
            "\\bcallable\\b",
            "\\bchr\\b",
            "\\bclassmethod\\b",
            "\\bcompile\\b",
            "\\bcomplex\\b",
            "\\bdelattr\\b",
            "\\bdict\\b",
            "\\bdir\\b",
            "\\bdivmod\\b",
            "\\benumerate\\b",
            "\\beval\\b",
            "\\bexec\\b",
            "\\bfilter\\b",
            "\\bfloat\\b",
            "\\bformat\\b",
            "\\bfrozenset\\b",
            "\\bgetattr\\b",
            "\\bglobals\\b",
            "\\bhasattr\\b",
            "\\bhash\\b",
            "\\bhelp\\b",
            "\\bhex\\b",
            "\\bid\\b",
            "\\binput\\b",
            "\\bint\\b",
            "\\bisinstance\\b",
            "\\bissubclass\\b",
            "\\biter\\b",
            "\\blen\\b",
            "\\blist\\b",
            "\\blocals\\b",
            "\\bmap\\b",
            "\\bmax\\b",
            "\\bmemoryview\\b",
            "\\bmin\\b",
            "\\bnext\\b",
            "\\bobject\\b",
            "\\boct\\b",
            "\\bopen\\b",
            "\\bord\\b",
            "\\bpow\\b",
            "\\bprint\\b",
            "\\bproperty\\b",
            "\\brange\\b",
            "\\brepr\\b",
            "\\breversed\\b",
            "\\bround\\b",
            "\\bset\\b",
            "\\bsetattr\\b",
            "\\bslice\\b",
            "\\bsorted\\b",
            "\\bstaticmethod\\b",
            "\\bstr\\b",
            "\\bsum\\b",
            "\\bsuper\\b",
            "\\btuple\\b",
            "\\btype\\b",
            "\\bvars\\b",
            "\\bzip\\b",
        ]
        self.highlighting_rules.extend(
            [(QRegularExpression(pattern), builtin_format) for pattern in builtins]
        )

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#bd93f9"))
        self.highlighting_rules.append(
            (QRegularExpression("\\b[0-9]+\\.?[0-9]*\\b"), number_format)
        )

        # Strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#f1fa8c"))
        self.highlighting_rules.append((QRegularExpression('".*"'), string_format))
        self.highlighting_rules.append((QRegularExpression("'.*'"), string_format))

        self.tri_single = (QRegularExpression("'''"), 1, string_format)
        self.tri_double = (QRegularExpression('"""'), 2, string_format)

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6272a4"))
        self.highlighting_rules.append((QRegularExpression("#[^\n]*"), comment_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)

        self.setCurrentBlockState(0)
        start_index = 0
        if self.previousBlockState() != 1:
            start_index = text.find("'''")
        if self.previousBlockState() != 2:
            start_index = text.find('"""')
        while start_index >= 0:
            end_index = text.find(
                "'''" if self.previousBlockState() != 2 else '"""', start_index + 3
            )
            if end_index == -1:
                self.setCurrentBlockState(1 if self.previousBlockState() != 2 else 2)
                comment_len = len(text) - start_index
            else:
                comment_len = end_index - start_index + 3
            self.setFormat(
                start_index,
                comment_len,
                self.tri_single[2] if self.previousBlockState() != 2 else self.tri_double[2],
            )
            start_index = text.find(
                "'''" if self.previousBlockState() != 2 else '"""', start_index + comment_len
            )


def create_icon_from_svg(svg_content: str, color: str) -> QIcon:
    """Create a QIcon from SVG content, with a specific color."""
    svg_content = svg_content.replace("currentColor", color)
    renderer = QSvgRenderer(svg_content.encode("utf-8"))
    pixmap = QPixmap(renderer.defaultSize())
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


class CodeBlockWidget(QFrame):
    """A widget to display a block of code with syntax highlighting and a copy button."""

    SVG_COPY = """
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard" viewBox="0 0 16 16">
      <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
      <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
    </svg>
    """
    SVG_CHECK = """
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-check-lg" viewBox="0 0 16 16">
      <path d="M12.736 3.97a.733.733 0 0 1 1.047 0c.286.289.29.756.01 1.05L7.88 12.01a.733.733 0 0 1-1.065.02L3.217 8.384a.757.757 0 0 1 0-1.06.733.733 0 0 1 1.047 0l3.052 3.093 5.4-6.425a.247.247 0 0 1 .02-.022z"/>
    </svg>
    """

    def __init__(self, language: str, code: str, parent=None):
        super().__init__(parent)
        self.setObjectName("codeBlock")
        self.language = language
        self.code = code

        self.icon_copy = create_icon_from_svg(self.SVG_COPY, "#f8f8f2")
        self.icon_check = create_icon_from_svg(self.SVG_CHECK, "#50fa7b")

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        header = QFrame(self)
        header.setObjectName("codeBlockHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)

        self.lang_label = QLabel(self.language.upper(), self)
        self.lang_label.setObjectName("codeBlockLangLabel")

        self.copy_button = QPushButton(self)
        self.copy_button.setIcon(self.icon_copy)
        self.copy_button.setObjectName("codeCopyButton")
        self.copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_button.clicked.connect(self.copy_code)

        header_layout.addWidget(self.lang_label)
        header_layout.addStretch()
        header_layout.addWidget(self.copy_button)

        self.code_edit = QTextEdit(self)
        self.code_edit.setPlainText(self.code.strip())
        self.code_edit.setReadOnly(True)
        self.code_edit.setObjectName("codeBlockEdit")
        self.code_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.code_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.highlighter = PythonSyntaxHighlighter(self.code_edit.document())

        layout.addWidget(header)
        layout.addWidget(self.code_edit)

        self.code_edit.document().documentLayout().documentSizeChanged.connect(self.update_height)
        QTimer.singleShot(0, self.update_height)

    def update_height(self):
        doc_height = self.code_edit.document().size().height()
        margins = self.code_edit.contentsMargins()
        buffer = 5
        new_height = doc_height + margins.top() + margins.bottom() + buffer
        self.code_edit.setFixedHeight(int(new_height))

    def copy_code(self):
        """Copy the code to the clipboard and show a temporary confirmation."""
        QApplication.clipboard().setText(self.code.strip())
        self.copy_button.setIcon(self.icon_check)
        QTimer.singleShot(2000, lambda: self.copy_button.setIcon(self.icon_copy))
