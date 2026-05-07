from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QCheckBox,
)
from PyQt6.QtCore import Qt
from tabs.settings import load_settings, save_settings, apply_theme
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QApplication


class Tab4Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        # 左上にタイトルを置く
        title = QLabel('設定')
        layout.addWidget(title)

        # ダークモード切替（チェックボックスのみ）
        self.dark_checkbox = QCheckBox('ダークモード')
        self.dark_checkbox.stateChanged.connect(self._on_theme_changed)
        layout.addWidget(self.dark_checkbox)

        # 初期状態を設定ファイルから読み込む
        settings = load_settings()
        is_dark = bool(settings.get('dark', False))
        self.dark_checkbox.setChecked(is_dark)
        # 起動時にテーマを適用
        apply_theme(QApplication.instance(), is_dark)

        # スペース埋め
        layout.addStretch()
        self.setLayout(layout)

    def _on_theme_changed(self, state):
        # Use current checkbox state to decide theme (robust against signal arg types)
        checked = self.dark_checkbox.isChecked()
        self._apply_theme(checked)
        # 保存
        settings = load_settings()
        settings['dark'] = bool(checked)
        save_settings(settings)

    def _apply_theme(self, dark: bool):
        # Delegate actual application to settings.apply_theme to keep styles consistent
        apply_theme(QApplication.instance(), dark)
        # Ensure checkbox reflects current state without re-triggering handlers
        self.dark_checkbox.blockSignals(True)
        self.dark_checkbox.setChecked(bool(dark))
        self.dark_checkbox.blockSignals(False)
