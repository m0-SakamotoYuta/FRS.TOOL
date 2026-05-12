import sys
from PyQt6.QtWidgets import (
	QApplication,
	QMainWindow,
	QWidget,
	QVBoxLayout,
	QTabWidget,
)
from tabs.tab1 import Tab1Widget
from tabs.tab2 import Tab2Widget
from tabs.tab3 import Tab3Widget
from tabs.tab4 import Tab4Widget
from splash import Splash
from tabs.settings import load_settings, apply_theme


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle('FRS Simulator - Main GUI')
		self.resize(800, 600)

		self.tabs = QTabWidget()
		self.tab1 = Tab1Widget()
		self.tab2 = Tab2Widget()
		self.tabs.addTab(self.tab1, 'Ver. 1')
		self.tabs.addTab(self.tab2, 'Ver. 2（実用）')
		self.tabs.addTab(Tab3Widget(), 'タブ3')
		self.tabs.addTab(Tab4Widget(), '設定')
		self.tabs.setCurrentWidget(self.tab2)
		if hasattr(self.tab2, 'posture_subtabs'):
			self.tab2.posture_subtabs.setCurrentIndex(0)
		self.setCentralWidget(self.tabs)

	def closeEvent(self, event):
		for widget in (getattr(self, 'tab1', None), getattr(self, 'tab2', None)):
			cleanup = getattr(widget, 'cleanup', None)
			if callable(cleanup):
				try:
					cleanup()
				except Exception:
					pass
		super().closeEvent(event)


def main():
	app = QApplication(sys.argv)
	# まずスプラッシュ（ロード画面）を表示し、キャッシュ（設定）を読み込む
	splash = Splash()
	splash.show()
	# allow the splash to render
	app.processEvents()

	# 設定を読み込み・テーマを適用
	try:
		splash.append_log('設定ファイル読み込みを開始します...')
		settings = load_settings()
		splash.append_log('設定ファイル読み込み完了')
		apply_theme(app, bool(settings.get('dark', False)))
		splash.append_log('テーマを適用しました')
	except Exception as e:
		splash.append_log(f'設定読み込み中にエラー: {e}')

	# 準備ができたらメインウィンドウを作成してスプラッシュを閉じる
	win = MainWindow()
	# center splash relative to main window geometry before showing main
	screen = app.primaryScreen()
	if screen:
		# center splash on primary screen
		geo = screen.availableGeometry()
		sx = geo.x() + (geo.width() - splash.width()) // 2
		sy = geo.y() + (geo.height() - splash.height()) // 2
		splash.move(sx, sy)

	win.showMaximized()
	splash.close()
	sys.exit(app.exec())


if __name__ == '__main__':
	main()

