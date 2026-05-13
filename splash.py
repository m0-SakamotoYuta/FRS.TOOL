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
        # 角丸の不透明な背景にして、緑のプログレスバーが映えるようにする
        self.setFixedSize(440, 240)
        self.setStyleSheet(
            'QWidget#splashRoot { background-color: #1f242c; border: 1px solid #3a4150; border-radius: 8px; } '
            'QLabel { color: #eaeaea; font-size: 13px; } '
            'QTextEdit { background-color: #14171c; color: #d6d6d6; border: 1px solid #2a2f38; border-radius: 4px; }'
        )
        self.setObjectName('splashRoot')

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        self.label = QLabel('FRS Tools\n読み込み中...')
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        # 緑色のプログレスバー（既定はアニメーション動作の不定モード）
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(18)
        self.progress.setStyleSheet(
            'QProgressBar { '
            '    background-color: #14171c; '
            '    border: 1px solid #2a2f38; '
            '    border-radius: 4px; '
            '} '
            'QProgressBar::chunk { '
            '    background-color: #4caf50; '
            '    border-radius: 3px; '
            '}'
        )
        layout.addWidget(self.progress)

        # ログ表示エリア
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(80)
        layout.addWidget(self.log)
        self.setLayout(layout)

    def append_log(self, text: str):
        self.log.append(text)

    def set_progress(self, value: int):
        """0–100 の決定モードへ切替。0 未満は不定（マーキー）モード。"""
        if value is None or value < 0:
            self.progress.setRange(0, 0)
        else:
            v = max(0, min(100, int(value)))
            self.progress.setRange(0, 100)
            self.progress.setValue(v)
            self.progress.setTextVisible(True)


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
