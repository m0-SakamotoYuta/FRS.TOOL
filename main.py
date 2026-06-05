import sys
import os  # 👈 追加
os.environ["QT_QUICK_CONTROLS_STYLE"] = "Basic"  # 👈 追加（OSのスタイル適用を無効化）
from PyQt6.QtWidgets import (
	QApplication,
	QMainWindow,
	QWidget,
	QVBoxLayout,
	QTabWidget,
	QAbstractSpinBox,
)
from PyQt6.QtCore import QObject, QEvent
from tabs.tab1 import Tab1Widget
from tabs.tab2 import Tab2Widget, FEAxisWidget, InitialPostureCandidatesWidget
from tabs.tab3 import Tab3Widget
from tabs.tab4 import Tab4Widget


class _NoWheelOnSpinBoxFilter(QObject):
	"""QSpinBox / QDoubleSpinBox 上でマウスホイールによる値変更を無効化する。
	右側の▲▼ボタンか、直接入力でのみ値を変更できるようにする。"""

	def eventFilter(self, watched, event):
		if event.type() == QEvent.Type.Wheel and isinstance(watched, QAbstractSpinBox):
			event.ignore()
			return True
		return False
from splash import Splash
from tabs.settings import load_settings, apply_theme, ensure_startup_backup


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle('FRS Tools - Main GUI')
		self.resize(800, 600)

		self.tabs = QTabWidget()
		self.tab1 = Tab1Widget()
		self.tab2 = Tab2Widget()
		self.tab_candidates = InitialPostureCandidatesWidget(self.tab2)
		self.tab_fe = FEAxisWidget()
		self.tabs.addTab(self.tab1, '初期姿勢校正Ver. 1（色変更）')
		self.tabs.addTab(self.tab2, '初期姿勢校正Ver. 2（実用）')
		self.tabs.addTab(self.tab_candidates, '初期姿勢候補')
		self.tabs.addTab(self.tab_fe, '呂')
		self.tabs.addTab(Tab3Widget(), 'KKR graph')
		self.tabs.addTab(Tab4Widget(), '設定')
		self.tabs.setCurrentWidget(self.tab2)
		if hasattr(self.tab2, 'posture_subtabs'):
			self.tab2.posture_subtabs.setCurrentIndex(0)
		self.setCentralWidget(self.tabs)

	def closeEvent(self, event):
		for widget in (
			getattr(self, 'tab1', None),
			getattr(self, 'tab2', None),
			getattr(self, 'tab_fe', None),
		):
			cleanup = getattr(widget, 'cleanup', None)
			if callable(cleanup):
				try:
					cleanup()
				except Exception:
					pass
		super().closeEvent(event)


def main():
	app = QApplication(sys.argv)
	# スピンボックスのマウスホイール無効化（アプリ全体に適用）
	_spin_filter = _NoWheelOnSpinBoxFilter()
	app.installEventFilter(_spin_filter)
	app._spin_filter_ref = _spin_filter   # GC 防止のため参照保持

	# まずスプラッシュ（ロード画面）を表示し、キャッシュ（設定）を読み込む
	splash = Splash()

	# 画面中央へ配置してから表示
	screen = app.primaryScreen()
	if screen:
		geo = screen.availableGeometry()
		sx = geo.x() + (geo.width() - splash.width()) // 2
		sy = geo.y() + (geo.height() - splash.height()) // 2
		splash.move(sx, sy)

	splash.show()
	splash.set_progress(5)
	splash.append_log('起動を開始します...')
	app.processEvents()

	# 設定を読み込み・テーマを適用
	try:
		splash.append_log('設定ファイル読み込みを開始します...')
		app.processEvents()
		# 起動時バックアップ（settings.json.startup と日付付きバックアップ）
		try:
			ensure_startup_backup()
			splash.append_log('起動時バックアップを作成しました')
			app.processEvents()
		except Exception as e:
			splash.append_log(f'起動時バックアップ作成中にエラー: {e}')
			app.processEvents()
		settings = load_settings()
		splash.set_progress(25)
		splash.append_log('設定ファイル読み込み完了')
		app.processEvents()
		apply_theme(app, bool(settings.get('dark', False)))
		splash.set_progress(40)
		splash.append_log('テーマを適用しました')
		app.processEvents()
	except Exception as e:
		splash.append_log(f'設定読み込み中にエラー: {e}')
		app.processEvents()

	# 準備ができたらメインウィンドウを作成
	splash.append_log('メインウィンドウを構築中（タブ・3D ビュー初期化）...')
	splash.set_progress(50)
	app.processEvents()
	win = MainWindow()
	splash.set_progress(90)
	splash.append_log('メインウィンドウ構築完了')
	app.processEvents()

	win.showMaximized()
	splash.set_progress(100)
	splash.append_log('準備完了')
	app.processEvents()
	splash.close()
	sys.exit(app.exec())


if __name__ == '__main__':
	main()

