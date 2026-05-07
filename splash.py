from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QDialog,
    QTextEdit,
    QProgressBar,
    QPushButton,
)
from PyQt6.QtCore import Qt


class Splash(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(400, 200)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        self.label = QLabel('FRS Simulator\n読み込み中...')
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        # simple log area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(70)
        layout.addWidget(self.log)
        self.setLayout(layout)

    def append_log(self, text: str):
        self.log.append(text)


class LoadingDialog(QDialog):
    """Reusable loading dialog with log and progress bar."""
    def __init__(self, title: str = '読み込み中', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(500, 300)
        layout = QVBoxLayout()
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate by default
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.cancel_btn = QPushButton('閉じる')
        self.cancel_btn.clicked.connect(self.close)
        layout.addWidget(self.cancel_btn)

        self.setLayout(layout)

    def append_log(self, text: str):
        self.log.append(text)

    def set_progress_indeterminate(self, indeterminate: bool = True):
        if indeterminate:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
