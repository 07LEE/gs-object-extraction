"""Small controls for the side panel."""

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class Choice(QObject):
    """Two named states behind a checkable button, with the calls of a combo box: the button is the control, this is how the window reads it."""

    currentIndexChanged = Signal(int)

    def __init__(self, button, names, parent=None):
        super().__init__(parent)
        self._button, self._names = button, tuple(names)
        button.toggled.connect(lambda on: self.currentIndexChanged.emit(int(on)))

    def currentIndex(self):
        return int(self._button.isChecked())

    def currentText(self):
        return self._names[self.currentIndex()]

    def setCurrentIndex(self, index):
        self._button.setChecked(bool(index))

    def setCurrentText(self, text):
        self.setCurrentIndex(self._names.index(text))

    def isEnabled(self):
        return self._button.isEnabled()

    def setEnabled(self, on):
        self._button.setEnabled(on)


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
