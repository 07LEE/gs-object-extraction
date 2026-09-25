"""Small controls for the side panel."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QToolButton, QVBoxLayout, QWidget


class Segmented(QWidget):
    """A row of buttons of which exactly one is down; the same calls as a combo box, for a choice that is changed often."""

    currentIndexChanged = Signal(int)

    def __init__(self, items, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._buttons = []
        for index, text in enumerate(items):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setStyleSheet("QPushButton:checked { background: palette(highlight); color: palette(highlighted-text); }")
            self._group.addButton(button, index)
            layout.addWidget(button, 1)
            self._buttons.append(button)
        self._buttons[0].setChecked(True)
        self._index = 0
        self._group.idClicked.connect(self.setCurrentIndex)

    def currentIndex(self):
        return self._index

    def currentText(self):
        return self._buttons[self._index].text()

    def setCurrentIndex(self, index):
        self._buttons[index].setChecked(True)
        if index != self._index:
            self._index = index
            self.currentIndexChanged.emit(index)

    def setCurrentText(self, text):
        for index, button in enumerate(self._buttons):
            if button.text() == text:
                self.setCurrentIndex(index)


class Collapsible(QWidget):
    """A heading that opens and closes the controls under it."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.RightArrow)
        self._toggle.setStyleSheet("QToolButton { border: none; }")
        self._body = QWidget()
        self._body.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toggle)
        layout.addWidget(self._body)
        self._toggle.toggled.connect(self._open)

    def body(self):
        return self._body

    def _open(self, on):
        self._toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self._body.setVisible(on)
