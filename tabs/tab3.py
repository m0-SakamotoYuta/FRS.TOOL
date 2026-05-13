from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class Tab3Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label = QLabel('タブ3 のプレースホルダ.将来的には、KKR graphや関節面接触動態可視化プログラムをここに追加したい。可視化プログラムはかなり大きいが、それは現実的か？')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)
