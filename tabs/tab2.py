from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QGroupBox, QRadioButton,
    QButtonGroup, QScrollArea, QPushButton, QFileDialog, QTextEdit, QListWidget,
    QSlider, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QColor
import os
import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

from tabs.settings import load_settings, save_settings


class DnDWidget(QWidget):
    """STL ファイルのドラッグ&ドロップを受け付ける QWidget。

    使い方: 生成後に `set_dnd_callback(callback)` を呼ぶ。callback は path: str を受け取る。
    .stl ファイル以外はドロップ拒否。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._dnd_callback = None

    def set_dnd_callback(self, callback):
        self._dnd_callback = callback

    def _extract_stl_paths(self, mime_data):
        if not mime_data.hasUrls():
            return []
        paths = []
        for url in mime_data.urls():
            p = url.toLocalFile()
            if p and p.lower().endswith('.stl') and os.path.isfile(p):
                paths.append(p)
        return paths

    def dragEnterEvent(self, event):
        if self._extract_stl_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._extract_stl_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = self._extract_stl_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        # 1 ファイルだけ採用（最初のもの）
        cb = self._dnd_callback
        if callable(cb):
            try:
                cb(paths[0])
            except Exception:
                pass
        event.acceptProposedAction()


class STLLoadWorker(QObject):
    """STL ロード用ワーカースレッド"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        if not HAS_PYVISTA:
            self.error.emit('pyvista または pyvistaqt が見つかりません。')
            return
        try:
            self.log.emit('STL読み込みを開始します...')
            mesh = pv.read(self.path)
            mesh = mesh.extract_surface().triangulate()
            self.log.emit('法線を計算します...')
            mesh = mesh.compute_normals(
                cell_normals=False,
                point_normals=True,
                split_vertices=True,
                consistent_normals=True,
                auto_orient_normals=True,
                non_manifold_traversal=True,
                feature_angle=45.0,
            )
            self.log.emit(f'読み込み完了: points={mesh.n_points}, cells={mesh.n_cells}')
            self.finished.emit(mesh)
        except Exception as e:
            self.error.emit(f'STL読み込み失敗: {e}')


class Tab2Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 軸ごとに独立した posture_widgets を保持: {'u': {...}, 'v': {...}, 'w': {...}}
        self.axis_data = {}
        self.visual_widgets = []
        layout = QVBoxLayout(self)

        subtabs = QTabWidget()
        axis_names = ['ALL VIEW', 'U axis', 'V axis', 'W axis', 'X axis', 'Y axis', 'Z axis']
        for axis_name in axis_names:
            if axis_name == 'ALL VIEW':
                axis_widget = self._create_all_view_tab()
            elif axis_name in ('U axis', 'V axis', 'W axis'):
                letter = axis_name[0].lower()  # 'u' / 'v' / 'w'
                axis_widget = self._create_axis_tab(letter, joint_type='rotation')
            elif axis_name in ('X axis', 'Y axis', 'Z axis'):
                letter = axis_name[0].lower()  # 'x' / 'y' / 'z'
                axis_widget = self._create_axis_tab(letter, joint_type='translation')
            else:
                axis_widget = self._create_simple_axis_tab(axis_name)
            subtabs.addTab(axis_widget, axis_name)

        layout.addWidget(subtabs)
        self.setLayout(layout)

    def _load_visual_settings(self):
        settings = load_settings() or {}
        tab1 = settings.get('tab1') or {}

        background_color = tab1.get('background_color', '#2a2f38')
        model_color = tab1.get('model_color', '#d9dbe0')

        if not isinstance(background_color, str) or not QColor(background_color).isValid():
            background_color = '#2a2f38'
        if not isinstance(model_color, str) or not QColor(model_color).isValid():
            model_color = '#d9dbe0'

        return QColor(background_color).name(), QColor(model_color).name()

    def _background_top_color(self, background_color: str):
        base = QColor(background_color)
        if not base.isValid():
            return '#12161d'
        return base.darker(190).name()

    def _apply_posture_visual_settings(self, widget):
        if not HAS_PYVISTA or widget is None or getattr(widget, 'plotter', None) is None:
            return

        background_color, model_color = self._load_visual_settings()
        widget.plotter.set_background(background_color, top=self._background_top_color(background_color))

        try:
            actor = widget.plotter.renderer.actors.get('stl_model')
        except Exception:
            actor = None

        if actor is not None:
            color = QColor(model_color)
            if color.isValid():
                actor.GetProperty().SetColor(color.redF(), color.greenF(), color.blueF())

    def _refresh_all_posture_visuals(self):
        for widget in self.visual_widgets:
            self._apply_posture_visual_settings(widget)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_all_posture_visuals()

    def cleanup(self):
        for widget in self.visual_widgets:
            plotter = getattr(widget, 'plotter', None)
            if plotter is None:
                continue
            try:
                plotter.close()
            except Exception:
                pass
            try:
                plotter.deleteLater()
            except Exception:
                pass
            widget.plotter = None

            thread = getattr(widget, 'thread', None)
            if thread is not None:
                try:
                    thread.quit()
                    thread.wait(1000)
                except Exception:
                    pass

    def _configure_lights(self, plotter):
        if plotter is None or not HAS_PYVISTA:
            return
        plotter.remove_all_lights()

        key = pv.Light(position=(3.0, 2.0, 2.5), focal_point=(0.0, 0.0, 0.0), color='white', intensity=1.0)
        fill = pv.Light(position=(-2.0, -1.5, 1.5), focal_point=(0.0, 0.0, 0.0), color='#cfd7ff', intensity=0.45)
        rim = pv.Light(position=(-1.5, 2.5, -2.0), focal_point=(0.0, 0.0, 0.0), color='#fff1d6', intensity=0.35)

        plotter.add_light(key)
        plotter.add_light(fill)
        plotter.add_light(rim)
    
    def _create_simple_axis_tab(self, axis_name: str) -> QWidget:
        """シンプルなプレースホルダータブを作成。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 軸タイトル
        title = QLabel(f'{axis_name}')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        layout.addWidget(title)
        
        # プレースホルダーラベル
        placeholder_label = QLabel(f'{axis_name} (プレースホルダ)')
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder_label)
        
        layout.addStretch()
        
        return widget
    
    def _create_all_view_tab(self) -> QWidget:
        """ALL VIEW タブ：base STL + C_world + U/V/W rotation-axis + X/Y/Z parallel-axis の重ね合わせ表示。"""
        widget = DnDWidget()
        widget.posture_key = 'main'
        widget.axis_letter = 'base'
        widget.c_axis_label_prefix = ''
        widget.c_axis_name = ''
        main_layout = QVBoxLayout(widget)

        top_layout = QHBoxLayout()

        # === 左パネル ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel('ALL VIEW (Base 姿勢)')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        left_layout.addWidget(title)

        load_btn = QPushButton('STL(base) を読み込む')
        left_layout.addWidget(load_btn)

        build_world_btn = QPushButton('C_world 座標系を生成')
        build_world_btn.setEnabled(False)
        left_layout.addWidget(build_world_btn)

        clear_world_btn = QPushButton('C_world 座標系を消去')
        clear_world_btn.setEnabled(False)
        left_layout.addWidget(clear_world_btn)

        import_rotation_btn = QPushButton('U/V/W rotation-axis を取り込み / 更新')
        import_rotation_btn.setEnabled(False)
        left_layout.addWidget(import_rotation_btn)

        import_parallel_btn = QPushButton('X/Y/Z parallel-axis を取り込み / 更新')
        import_parallel_btn.setEnabled(False)
        left_layout.addWidget(import_parallel_btn)

        # === 軸の表示切替 ===
        visibility_group = QGroupBox('軸の表示')
        vis_layout = QVBoxLayout(visibility_group)
        # ヘッダー
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel('  軸'), 1)
        header_row.addWidget(QLabel('矢印'))
        vis_layout.addLayout(header_row)

        axis_check_specs = [
            ('U_rotation-axis', '#ffd000'),
            ('V_rotation-axis', '#00d4d4'),
            ('W_rotation-axis', '#ff60a0'),
            ('X_parallel-axis', '#ff8040'),
            ('Y_parallel-axis', '#80ff40'),
            ('Z_parallel-axis', '#4080ff'),
        ]
        axis_checkboxes = {}
        axis_arrow_checkboxes = {}
        for name, color in axis_check_specs:
            row = QHBoxLayout()
            cb_main = QCheckBox(name)
            cb_main.setChecked(True)
            cb_main.setStyleSheet(f'color: {color}; font-weight: bold;')
            cb_arrow = QCheckBox()
            cb_arrow.setChecked(True)
            row.addWidget(cb_main, 1)
            row.addWidget(cb_arrow)
            vis_layout.addLayout(row)
            axis_checkboxes[name] = cb_main
            axis_arrow_checkboxes[name] = cb_arrow
        left_layout.addWidget(visibility_group)

        # === 検討事項 グループ ===
        check_group = QGroupBox('検討事項')
        check_layout = QVBoxLayout(check_group)

        check_label = QLabel('（軸を取り込むと結果がここに表示されます）')
        check_label.setWordWrap(True)
        check_label.setTextFormat(Qt.TextFormat.RichText)
        check_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        check_layout.addWidget(check_label)

        # 球サイズ スライダー + 数値入力
        sphere_size_row = QHBoxLayout()
        sphere_size_lbl = QLabel('球サイズ:')
        sphere_size_lbl.setMinimumWidth(80)
        sphere_size_slider = QSlider(Qt.Orientation.Horizontal)
        sphere_size_slider.setRange(1, 300)
        sphere_size_slider.setValue(50)
        sphere_size_spin = QSpinBox()
        sphere_size_spin.setRange(1, 300)
        sphere_size_spin.setValue(50)
        sphere_size_spin.setSuffix('%')
        sphere_size_spin.setMinimumWidth(80)
        # 双方向同期
        sphere_size_slider.valueChanged.connect(sphere_size_spin.setValue)
        sphere_size_spin.valueChanged.connect(sphere_size_slider.setValue)
        sphere_size_row.addWidget(sphere_size_lbl)
        sphere_size_row.addWidget(sphere_size_slider, 1)
        sphere_size_row.addWidget(sphere_size_spin)
        check_layout.addLayout(sphere_size_row)

        # 球透明度 スライダー + 数値入力
        sphere_opacity_row = QHBoxLayout()
        sphere_opacity_lbl = QLabel('球透明度:')
        sphere_opacity_lbl.setMinimumWidth(80)
        sphere_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        sphere_opacity_slider.setRange(0, 100)
        sphere_opacity_slider.setValue(60)
        sphere_opacity_spin = QSpinBox()
        sphere_opacity_spin.setRange(0, 100)
        sphere_opacity_spin.setValue(60)
        sphere_opacity_spin.setSuffix('%')
        sphere_opacity_spin.setMinimumWidth(80)
        sphere_opacity_slider.valueChanged.connect(sphere_opacity_spin.setValue)
        sphere_opacity_spin.valueChanged.connect(sphere_opacity_slider.setValue)
        sphere_opacity_row.addWidget(sphere_opacity_lbl)
        sphere_opacity_row.addWidget(sphere_opacity_slider, 1)
        sphere_opacity_row.addWidget(sphere_opacity_spin)
        check_layout.addLayout(sphere_opacity_row)

        left_layout.addWidget(check_group)

        # === 右パネル: 3D ビュー ===
        if HAS_PYVISTA:
            plotter = QtInteractor(widget)
            background_color, _ = self._load_visual_settings()
            plotter.set_background(background_color, top=self._background_top_color(background_color))
            plotter.add_text('STL(base) を読み込んでください', position='upper_left', font_size=10)
            self._configure_lights(plotter)
            right_view = plotter.interactor
        else:
            plotter = None
            right_view = QLabel('pyvista / pyvistaqt が未インストール')
            right_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        widget.plotter = plotter

        # === 下段: ログ ===
        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setPlaceholderText('ログ')
        log_view.setMinimumHeight(120)
        log_view.setMaximumHeight(220)

        # === 平面サブタブ (W のみ) ===
        plane_subtabs = QTabWidget()
        widget.plane_subtabs = plane_subtabs
        plane_specs = [
            ('W平面1（XY平面）', 'W平面1（XY平面） [C_world 用]'),
            ('W平面2（YZ平面）', 'W平面2（YZ平面） [C_world 用]'),
            ('W平面3（ZX平面）', 'W平面3（ZX平面） [C_world 用]'),
        ]

        plane_widgets = []
        widget.shared_points = {}        # 未使用（C_local は無い）
        widget.shared_world_points = {}
        widget.active_plane_index = 0

        for plane_label, plane_title in plane_specs:
            plane_widget = self._create_plane_point_controls_widget(plane_label, plane_title)
            plane_widget.posture_widget = widget
            plane_widget.plotter = plotter
            plane_widget.log_view = log_view
            plane_widget.system_type = 'c_world'
            plane_widget.points = widget.shared_world_points.setdefault(plane_label, [])
            self._wire_plane_point_handlers(plane_widget)
            plane_subtabs.addTab(plane_widget, plane_label)
            plane_widgets.append(plane_widget)

        left_layout.addWidget(plane_subtabs, 1)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(280)
        left_scroll.setMaximumWidth(420)

        top_layout.addWidget(left_scroll, 1)
        top_layout.addWidget(right_view, 4)

        main_layout.addLayout(top_layout, 5)
        main_layout.addWidget(log_view, 1)

        # === 状態保持 ===
        widget.load_btn = load_btn
        widget.build_world_btn = build_world_btn
        widget.clear_world_btn = clear_world_btn
        widget.import_rotation_btn = import_rotation_btn
        widget.import_parallel_btn = import_parallel_btn
        widget.axis_checkboxes = axis_checkboxes
        widget.axis_arrow_checkboxes = axis_arrow_checkboxes
        widget.axis_visibility = {name: True for name, _ in axis_check_specs}
        widget.axis_arrow_visibility = {name: True for name, _ in axis_check_specs}
        widget.check_label = check_label
        widget.sphere_size_slider = sphere_size_slider
        widget.sphere_size_spin = sphere_size_spin
        widget.sphere_opacity_slider = sphere_opacity_slider
        widget.sphere_opacity_spin = sphere_opacity_spin
        widget.log_view = log_view
        widget.plane_widgets = plane_widgets
        widget.current_mesh = None
        widget.c_axis = None
        widget.c_world = None
        widget.stl_path = None
        widget.rotation_axes = None       # U/V/W rotation-axes
        widget.parallel_axes = None       # X/Y/Z parallel-axes
        widget.intersection_point = None  # C_world 上の rotation 最近接 / 交点
        widget.intersection_distances = None
        widget.parallel_pair_angles = None  # [(name1, name2, angle_deg), ...]
        # _build_c_axis/clear_c_axis 等の互換用ダミー（実際には使わない）
        widget.build_axis_btn = QPushButton(); widget.build_axis_btn.hide()
        widget.clear_axis_btn = QPushButton(); widget.clear_axis_btn.hide()
        self.visual_widgets.append(widget)

        # === STL ロード／メッシュ反映 ===
        def _start_load(path: str, preserve_state: bool = False):
            widget._next_load_preserve = preserve_state
            widget._pending_load_path = path
            widget.log_view.setText('')
            widget.log_view.append(
                f'読込要求: {path}' + ('（キャッシュ復元）' if preserve_state else '')
            )
            widget.load_btn.setEnabled(False)

            widget.thread = QThread(widget)
            widget.worker = STLLoadWorker(path)
            widget.worker.moveToThread(widget.thread)
            widget.thread.started.connect(widget.worker.run)
            widget.worker.log.connect(lambda msg: widget.log_view.append(msg))
            widget.worker.finished.connect(lambda mesh: _on_mesh_loaded(mesh))
            widget.worker.error.connect(lambda msg: _on_load_error(msg))
            widget.worker.finished.connect(widget.thread.quit)
            widget.worker.error.connect(widget.thread.quit)
            widget.worker.finished.connect(widget.worker.deleteLater)
            widget.worker.error.connect(widget.worker.deleteLater)
            widget.thread.finished.connect(widget.thread.deleteLater)
            widget.thread.start()

        def _on_mesh_loaded(mesh):
            preserve = bool(getattr(widget, '_next_load_preserve', False))
            widget._next_load_preserve = False
            widget.current_mesh = mesh
            widget.stl_path = getattr(widget, '_pending_load_path', None) or getattr(widget, 'stl_path', None)
            widget.log_view.append('表示を更新します...')

            if not preserve:
                widget.c_world = None
                widget.rotation_axes = None
                widget.parallel_axes = None
                widget.intersection_point = None
                widget.intersection_distances = None
                widget.parallel_pair_angles = None
                widget.check_label.setText('（軸を取り込むと結果がここに表示されます）')
                for plane_widget in widget.plane_widgets:
                    plane_widget.current_mesh = mesh
                    plane_widget.points.clear()
                    plane_widget.selected_point_index = -1
                    plane_widget.point_add_enabled = False
                    plane_widget.point_add_btn.setChecked(False)
                    plane_widget.point_add_btn.setEnabled(True)
                    plane_widget._refresh_point_list()
            else:
                for plane_widget in widget.plane_widgets:
                    plane_widget.current_mesh = mesh
                    plane_widget.point_add_enabled = False
                    plane_widget.point_add_btn.setChecked(False)
                    plane_widget.point_add_btn.setEnabled(True)
                    if plane_widget.points:
                        plane_widget.selected_point_index = len(plane_widget.points) - 1
                    else:
                        plane_widget.selected_point_index = -1
                    plane_widget._refresh_point_list()

            self._render_all_view(widget, reset_view=True)
            widget.build_world_btn.setEnabled(True)
            widget.clear_world_btn.setEnabled(widget.c_world is not None)
            widget.import_rotation_btn.setEnabled(widget.c_world is not None)
            widget.import_parallel_btn.setEnabled(widget.c_world is not None)
            widget.log_view.append('完了')
            widget.load_btn.setEnabled(True)
            self._save_posture_cache(widget)

        def _on_load_error(msg: str):
            widget.log_view.append(msg)
            widget.load_btn.setEnabled(True)

        widget._start_load = _start_load
        widget._on_mesh_loaded = _on_mesh_loaded
        widget._on_load_error = _on_load_error

        # STL ファイルの D&D を受け付ける
        if isinstance(widget, DnDWidget):
            widget.set_dnd_callback(lambda p: widget._start_load(p))

        def _refresh_all_plane_views(reset_view: bool = False):
            self._render_all_view(widget, reset_view=reset_view)
        widget._refresh_all_plane_views = _refresh_all_plane_views

        def _on_plane_subtab_changed(index: int):
            widget.active_plane_index = index
            if index < 0 or index >= len(widget.plane_widgets):
                return
            if widget.plotter is None or widget.current_mesh is None:
                return
            active = widget.plane_widgets[index]
            try:
                widget.plotter.disable_picking()
            except Exception:
                pass
            if active.point_add_enabled:
                try:
                    widget.plotter.enable_surface_point_picking(
                        callback=lambda point, *_args: self._on_plane_surface_point_picked(active, point),
                        left_clicking=True, show_point=False, pickable_window=False,
                    )
                except Exception:
                    pass
            self._render_all_view(widget, reset_view=False)
        widget._on_plane_subtab_changed = _on_plane_subtab_changed
        plane_subtabs.currentChanged.connect(_on_plane_subtab_changed)

        # === ボタン接続 ===
        load_btn.clicked.connect(lambda: self._open_posture_file(widget))
        build_world_btn.clicked.connect(lambda: self._build_c_world_for_all_view(widget))
        clear_world_btn.clicked.connect(lambda: self._clear_c_world_for_all_view(widget))
        import_rotation_btn.clicked.connect(lambda: self._import_axes(widget, kind='rotation'))
        import_parallel_btn.clicked.connect(lambda: self._import_axes(widget, kind='translation'))

        sphere_size_slider.valueChanged.connect(
            lambda v: self._on_sphere_size_changed(widget, v)
        )
        sphere_opacity_slider.valueChanged.connect(
            lambda v: self._on_sphere_opacity_changed(widget, v)
        )

        for name, cb in axis_checkboxes.items():
            cb.toggled.connect(
                lambda checked, n=name, w=widget: self._on_axis_visibility_toggled(w, n, checked)
            )
        for name, cb in axis_arrow_checkboxes.items():
            cb.toggled.connect(
                lambda checked, n=name, w=widget: self._on_axis_arrow_visibility_toggled(w, n, checked)
            )

        # === キャッシュ復元 ===
        cached_stl_path = self._load_posture_cache(widget)
        for plane_widget in plane_widgets:
            plane_widget._refresh_point_list()
        if cached_stl_path and os.path.exists(cached_stl_path):
            widget.log_view.append(f'キャッシュ検出: {cached_stl_path}')
            widget._start_load(cached_stl_path, preserve_state=True)
        elif cached_stl_path:
            widget.log_view.append(f'前回のSTLが見つかりません: {cached_stl_path}')

        return widget

    def _build_c_world_for_all_view(self, widget):
        plane_keys = ['W平面1（XY平面）', 'W平面2（YZ平面）', 'W平面3（ZX平面）']
        pts = self._collect_plane_points(widget.shared_world_points, plane_keys)
        result = self._compute_axis_system(pts, widget.log_view, prefix='C_world')
        if result is None:
            return
        widget.c_world = result
        widget.clear_world_btn.setEnabled(True)
        widget.import_rotation_btn.setEnabled(True)
        widget.import_parallel_btn.setEnabled(True)
        self._render_all_view(widget, reset_view=False)
        self._save_posture_cache(widget)

    def _clear_c_world_for_all_view(self, widget):
        if getattr(widget, 'c_world', None) is None:
            return
        widget.c_world = None
        widget.rotation_axes = None
        widget.parallel_axes = None
        widget.intersection_point = None
        widget.intersection_distances = None
        widget.parallel_pair_angles = None
        widget.clear_world_btn.setEnabled(False)
        widget.import_rotation_btn.setEnabled(False)
        widget.import_parallel_btn.setEnabled(False)
        widget.check_label.setText('（軸を取り込むと結果がここに表示されます）')
        widget.log_view.append('C_world 座標系を消去しました。回転軸の取り込みもクリアしました。')
        self._render_all_view(widget, reset_view=False)
        self._save_posture_cache(widget)

    def _on_sphere_size_changed(self, widget, value):
        # QSpinBox が値を表示するため、ここでは再描画のみ
        self._render_all_view(widget, reset_view=False)

    def _on_sphere_opacity_changed(self, widget, value):
        self._render_all_view(widget, reset_view=False)

    def _on_axis_visibility_toggled(self, widget, axis_name, checked):
        if not hasattr(widget, 'axis_visibility'):
            widget.axis_visibility = {}
        widget.axis_visibility[axis_name] = bool(checked)
        self._render_all_view(widget, reset_view=False)

    def _on_axis_arrow_visibility_toggled(self, widget, axis_name, checked):
        if not hasattr(widget, 'axis_arrow_visibility'):
            widget.axis_arrow_visibility = {}
        widget.axis_arrow_visibility[axis_name] = bool(checked)
        self._render_all_view(widget, reset_view=False)

    def _fetch_motion_axis(self, letter, name, color, kind, log):
        """U/V/W/X/Y/Z それぞれのタブから motion_axis を取得。失敗時は None。"""
        axis_data = self.axis_data.get(letter)
        if axis_data is None:
            log.append(f'{name}: 軸データが見つかりません。')
            return None
        motion_widget = axis_data.get('motion_widget')
        if motion_widget is None:
            log.append(f'{name}: motion_widget が見つかりません。')
            return None
        mot = getattr(motion_widget, 'motion_axis', None)
        if mot is None:
            kind_jp = '回転軸' if kind == 'rotation' else '並進軸'
            log.append(f'{name}: {letter.upper()} axis タブで{kind_jp}が未計算です。')
            return None
        d = np.asarray(mot['direction'], dtype=float)
        p = np.asarray(mot['point'], dtype=float)
        log.append(
            f'{name} 取得: 方向=({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}), '
            f'通る点=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})'
        )
        return {'name': name, 'color': color, 'direction': d, 'point': p, 'kind': kind}

    def _import_axes(self, widget, kind: str):
        """kind='rotation' なら U/V/W、'translation' なら X/Y/Z を取り込み、分析・描画する。"""
        log = widget.log_view
        log.append('---')

        if widget.c_world is None:
            log.append('C_world が未生成です。先に C_world 座標系を生成してください。')
            return

        if kind == 'rotation':
            log.append('U/V/W rotation-axis を取り込みます...')
            spec_list = [
                ('u', 'U_rotation-axis', '#ffd000'),
                ('v', 'V_rotation-axis', '#00d4d4'),
                ('w', 'W_rotation-axis', '#ff60a0'),
            ]
            kind_jp = '回転軸'
        else:
            log.append('X/Y/Z parallel-axis を取り込みます...')
            spec_list = [
                ('x', 'X_parallel-axis', '#ff8040'),
                ('y', 'Y_parallel-axis', '#80ff40'),
                ('z', 'Z_parallel-axis', '#4080ff'),
            ]
            kind_jp = '並進軸'

        new_axes = []
        for letter, name, color in spec_list:
            entry = self._fetch_motion_axis(letter, name, color, kind, log)
            if entry is not None:
                new_axes.append(entry)

        if kind == 'rotation':
            widget.rotation_axes = new_axes if new_axes else None
            self._analyze_rotation_axes(widget, log)
        else:
            widget.parallel_axes = new_axes if new_axes else None
            self._analyze_parallel_axes(widget, log)

        self._update_check_label(widget)
        self._render_all_view(widget, reset_view=False)

    def _analyze_rotation_axes(self, widget, log):
        """rotation_axes の最近接点・距離を計算して widget に格納。"""
        axes = widget.rotation_axes or []
        if len(axes) < 2:
            widget.intersection_point = None
            widget.intersection_distances = None
            widget.pairwise_rotation = None
            if axes:
                log.append('1点交差の判定には 2 本以上の回転軸が必要です。')
            return

        sumM = np.zeros((3, 3))
        sumMp = np.zeros(3)
        for ax in axes:
            d = ax['direction'] / (np.linalg.norm(ax['direction']) or 1.0)
            M = np.eye(3) - np.outer(d, d)
            sumM += M
            sumMp += M @ ax['point']
        try:
            q = np.linalg.solve(sumM, sumMp)
        except np.linalg.LinAlgError:
            log.append('回転軸の交点計算に失敗（軸が平行か縮退）。')
            widget.intersection_point = None
            widget.intersection_distances = None
            widget.pairwise_rotation = None
            return

        distances = []
        for ax in axes:
            d_unit = ax['direction'] / np.linalg.norm(ax['direction'])
            diff = q - ax['point']
            perp = diff - np.dot(diff, d_unit) * d_unit
            distances.append(float(np.linalg.norm(perp)))

        pairwise = []
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                p1, d1 = axes[i]['point'], axes[i]['direction']
                p2, d2 = axes[j]['point'], axes[j]['direction']
                cross = np.cross(d1, d2)
                cn = np.linalg.norm(cross)
                if cn < 1e-9:
                    d1u = d1 / np.linalg.norm(d1)
                    diff = p2 - p1
                    par = diff - np.dot(diff, d1u) * d1u
                    dist = float(np.linalg.norm(par))
                else:
                    dist = float(abs(np.dot(p2 - p1, cross)) / cn)
                pairwise.append((axes[i]['name'], axes[j]['name'], dist))

        widget.intersection_point = q
        widget.intersection_distances = distances
        widget.pairwise_rotation = pairwise

        max_dist = max(distances) if distances else 0.0
        intersect = max_dist < 1.0
        log.append(
            f'→ {"回転軸は1点で交わります" if intersect else f"回転軸は1点で交わりません（最大誤差 {max_dist:.3f} mm）"}'
        )
        log.append(f'推定点 (C_world): ({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f})')

    def _analyze_parallel_axes(self, widget, log):
        """parallel_axes の方向ペア間角度を計算して widget に格納。直交ステージなら理想 90°。"""
        axes = widget.parallel_axes or []
        if len(axes) < 2:
            widget.parallel_pair_angles = None
            if axes:
                log.append('並進軸の解析には 2 本以上が必要です。')
            return

        pair_angles = []
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                d1 = axes[i]['direction'] / np.linalg.norm(axes[i]['direction'])
                d2 = axes[j]['direction'] / np.linalg.norm(axes[j]['direction'])
                cos_th = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cos_th)))
                pair_angles.append((axes[i]['name'], axes[j]['name'], angle))

        widget.parallel_pair_angles = pair_angles

        for n1, n2, a in pair_angles:
            log.append(f'  {n1} ↔ {n2} なす角: {a:.4f}° (直交ステージなら理想 90°)')

    def _update_check_label(self, widget):
        """rotation_axes / parallel_axes の現状を 検討事項 ラベルへ反映。"""
        html_lines = []

        # --- 回転軸セクション ---
        rot_axes = widget.rotation_axes or []
        if rot_axes:
            html_lines.append('<b>■ U/V/W rotation-axis</b>')
            distances = widget.intersection_distances
            q = widget.intersection_point
            pairwise = getattr(widget, 'pairwise_rotation', None)
            if distances and q is not None:
                max_dist = max(distances)
                tol = 1.0
                if max_dist < tol:
                    html_lines.append(
                        f'　<b>1点で交わる:</b> <span style="color:#90ee90">はい</span>'
                        f'（最大誤差 {max_dist:.3f} mm &lt; {tol} mm）'
                    )
                else:
                    html_lines.append(
                        f'　<b>1点で交わる:</b> <span style="color:#ffa07a">いいえ</span>'
                        f'（最大誤差 {max_dist:.3f} mm）'
                    )
                html_lines.append('　<b>各軸 ↔ 推定点 距離 [mm]:</b>')
                for ax, d in zip(rot_axes, distances):
                    html_lines.append(f'　　• {ax["name"]}: {d:.4f}')
                if pairwise:
                    html_lines.append('　<b>軸ペア間距離 [mm]:</b>')
                    for n1, n2, d in pairwise:
                        html_lines.append(f'　　• {n1} ↔ {n2}: {d:.4f}')
                html_lines.append('　<b>推定点 (C_world):</b>')
                html_lines.append(f'　　({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f})')
            else:
                html_lines.append('　（回転軸を 2 本以上取り込むと交点解析を表示）')

        # --- 並進軸セクション ---
        par_axes = widget.parallel_axes or []
        if par_axes:
            if html_lines:
                html_lines.append('')  # 空行で区切り
            html_lines.append('<b>■ X/Y/Z parallel-axis</b>')
            html_lines.append('　<b>各軸の方向 (C_world):</b>')
            for ax in par_axes:
                d = ax['direction']
                html_lines.append(
                    f'　　• {ax["name"]}: ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})'
                )
            pair_angles = getattr(widget, 'parallel_pair_angles', None)
            if pair_angles:
                html_lines.append('　<b>軸ペアのなす角 [°]（直交ステージなら理想 90°）:</b>')
                for n1, n2, a in pair_angles:
                    dev = abs(a - 90.0)
                    color = '#90ee90' if dev < 1.0 else '#ffa07a'
                    html_lines.append(
                        f'　　• {n1} ↔ {n2}: <span style="color:{color}">{a:.4f}°</span> '
                        f'(90°との差 {dev:.4f}°)'
                    )

        if not html_lines:
            widget.check_label.setText('（軸を取り込むと結果がここに表示されます）')
        else:
            widget.check_label.setText('<br>'.join(html_lines))

    def _render_all_view(self, widget, reset_view: bool = False):
        plotter = getattr(widget, 'plotter', None)
        if plotter is None or widget.current_mesh is None:
            return

        try:
            plotter.disable_picking()
        except Exception:
            pass

        camera_position = None
        if not reset_view:
            try:
                camera_position = plotter.camera_position
            except Exception:
                pass

        plotter.clear()
        bg, model_color = self._load_visual_settings()
        plotter.set_background(bg, top=self._background_top_color(bg))
        self._configure_lights(plotter)
        plotter.hide_axes()

        plotter.add_mesh(
            widget.current_mesh,
            name='stl_model',
            color=model_color,
            show_edges=False, smooth_shading=True,
            ambient=0.15, diffuse=0.75, specular=0.35, specular_power=25.0,
            pickable=True,
        )

        plane_colors = {
            'W平面1（XY平面）': '#ff8c00',
            'W平面2（YZ平面）': '#00d4d4',
            'W平面3（ZX平面）': '#d040d0',
        }
        for plane_label, points in widget.shared_world_points.items():
            if not points:
                continue
            arr = np.array(points, dtype=float)
            plotter.add_mesh(
                pv.PolyData(arr),
                name=f'plane_points::{plane_label}',
                color=plane_colors.get(plane_label, '#ffffff'),
                point_size=12, render_points_as_spheres=True, style='points',
                pickable=False, reset_camera=False, render=False,
            )

        # アクティブ平面の選択ハイライト
        active_index = getattr(widget, 'active_plane_index', 0)
        active_plane = None
        if 0 <= active_index < len(widget.plane_widgets):
            active_plane = widget.plane_widgets[active_index]
            pts = getattr(active_plane, 'points', [])
            sel = getattr(active_plane, 'selected_point_index', -1)
            if 0 <= sel < len(pts):
                sel_arr = np.array([pts[sel]], dtype=float)
                plotter.add_mesh(
                    pv.PolyData(sel_arr),
                    name='selected_point',
                    color='#ffff66',
                    point_size=18, render_points_as_spheres=True, style='points',
                    pickable=False, reset_camera=False, render=False,
                )

        # 平面サーフェス（半透明）
        for plane_label, points in widget.shared_world_points.items():
            if not points or len(points) < 3:
                continue
            try:
                plane = self._build_plane_from_points(points)
            except Exception:
                plane = None
            if plane is None:
                continue
            is_selected = (getattr(active_plane, 'plane_label', None) == plane_label)
            try:
                plotter.add_mesh(
                    plane,
                    name=f'plane_surface::{plane_label}',
                    color=plane_colors.get(plane_label, '#ffffff'),
                    opacity=0.45 if is_selected else 0.20,
                    pickable=False, reset_camera=False, render=False,
                    show_edges=False, smooth_shading=True,
                )
            except Exception:
                pass

        # シーン基準長
        b = widget.current_mesh.bounds
        diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])) or 100.0

        # C_world 座標軸（STL 座標系で描画）
        if widget.c_world is not None:
            axis_len = max(diag * 0.2, 1.0)
            origin = widget.c_world['origin']
            for suffix, vec, color in (
                ('_x', widget.c_world['ex'], '#ff3030'),
                ('_y', widget.c_world['ey'], '#3060ff'),
                ('_z', widget.c_world['ez'], '#30c030'),
            ):
                try:
                    arrow = pv.Arrow(start=origin, direction=vec, scale=axis_len,
                                     shaft_radius=0.015, tip_radius=0.045, tip_length=0.20)
                    plotter.add_mesh(arrow, name=f'c_world{suffix}', color=color,
                                     pickable=False, reset_camera=False, render=False)
                except Exception:
                    pass
            try:
                plotter.add_mesh(
                    pv.PolyData(np.array([origin], dtype=float)),
                    name='c_world_origin', color='#80c0ff',
                    point_size=14, render_points_as_spheres=True, style='points',
                    pickable=False, reset_camera=False, render=False,
                )
            except Exception:
                pass
            try:
                offs = axis_len * 0.10
                label_pos = np.asarray(origin, dtype=float) + np.array([offs, offs, offs])
                plotter.add_point_labels(
                    np.array([label_pos], dtype=float),
                    ['C_world'],
                    name='c_world_label',
                    font_size=14, text_color='#bfe4ff',
                    point_size=0, shape=None, always_visible=True,
                    pickable=False, reset_camera=False, render=False,
                )
            except Exception:
                pass

        # 取り込み済み回転軸 + 並進軸 を STL 座標系へ変換して描画
        rotation_axes = widget.rotation_axes or []
        parallel_axes = widget.parallel_axes or []
        all_imported_axes = list(rotation_axes) + list(parallel_axes)

        # 軸の表示/非表示フィルタ
        visibility = getattr(widget, 'axis_visibility', {})
        arrow_visibility = getattr(widget, 'axis_arrow_visibility', {})
        visible_axes = [ax for ax in all_imported_axes if visibility.get(ax['name'], True)]

        if visible_axes and widget.c_world is not None:
            R_w = np.column_stack([widget.c_world['ex'], widget.c_world['ey'], widget.c_world['ez']])
            O_w = np.asarray(widget.c_world['origin'], dtype=float)

            for entry in visible_axes:
                name = entry['name']
                color = entry['color']
                dir_w = np.asarray(entry['direction'], dtype=float)
                pt_w = np.asarray(entry['point'], dtype=float)
                dir_stl = R_w @ dir_w
                pt_stl = O_w + R_w @ pt_w

                line_half = diag * 0.9
                start_pt = pt_stl - dir_stl * line_half
                end_pt = pt_stl + dir_stl * line_half
                try:
                    line = pv.Line(start_pt, end_pt)
                    plotter.add_mesh(line, name=f'axis_{name}_line', color=color,
                                     line_width=4, pickable=False, reset_camera=False, render=False)
                except Exception:
                    pass
                # 矢印は per-axis トグルでさらに制御
                if arrow_visibility.get(name, True):
                    try:
                        arrow = pv.Arrow(start=pt_stl, direction=dir_stl, scale=diag * 0.30,
                                         shaft_radius=0.018, tip_radius=0.055, tip_length=0.20)
                        plotter.add_mesh(arrow, name=f'axis_{name}_arrow', color=color,
                                         pickable=False, reset_camera=False, render=False)
                    except Exception:
                        pass
                try:
                    label_pos = pt_stl + dir_stl * (diag * 0.40)
                    label_pos = label_pos + np.array([diag * 0.03, diag * 0.03, diag * 0.03])
                    plotter.add_point_labels(
                        np.array([label_pos], dtype=float),
                        [name],
                        name=f'axis_{name}_label',
                        font_size=14, text_color=color,
                        point_size=0, shape=None, always_visible=True,
                        pickable=False, reset_camera=False, render=False,
                    )
                except Exception:
                    pass

            # 球：rotation 軸の最近接点 / 交点（rotation_axes が 2 本以上の場合のみ）
            if widget.intersection_point is not None and rotation_axes:
                q_w = np.asarray(widget.intersection_point, dtype=float)
                q_stl = O_w + R_w @ q_w
                base_radius = diag * 0.03
                slider_factor = widget.sphere_size_slider.value() / 50.0
                radius = max(base_radius * slider_factor, 1e-3)
                opacity = widget.sphere_opacity_slider.value() / 100.0
                # 交わるかどうかで色を変える
                max_dist = max(widget.intersection_distances) if widget.intersection_distances else 0.0
                sphere_color = '#90ff80' if max_dist < 1.0 else '#ff80c0'
                try:
                    sphere = pv.Sphere(radius=radius, center=q_stl,
                                       theta_resolution=24, phi_resolution=16)
                    plotter.add_mesh(sphere, name='intersection_sphere',
                                     color=sphere_color, opacity=opacity,
                                     smooth_shading=True, pickable=False,
                                     reset_camera=False, render=False)
                except Exception:
                    pass

        if reset_view:
            plotter.reset_camera(bounds=widget.current_mesh.bounds)
        elif camera_position is not None:
            try:
                plotter.camera_position = camera_position
            except Exception:
                pass

        if active_plane is not None and active_plane.point_add_enabled:
            try:
                plotter.enable_surface_point_picking(
                    callback=lambda point, *_args: self._on_plane_surface_point_picked(active_plane, point),
                    left_clicking=True, show_point=False, pickable_window=False,
                )
            except Exception:
                pass

        for pw in widget.plane_widgets:
            if hasattr(pw, 'point_add_btn'):
                pw.point_add_btn.setEnabled(pw.current_mesh is not None)

        plotter.render()

    def _create_axis_tab(self, axis_letter: str, joint_type: str = 'rotation') -> QWidget:
        """U/V/W/X/Y/Z 軸の共通タブ作成。
        joint_type='rotation' → U/V/W（回転関節）
        joint_type='translation' → X/Y/Z（直動関節）
        """
        widget = QWidget()
        main_layout = QVBoxLayout(widget)

        title = QLabel(f'{axis_letter.upper()} axis')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        main_layout.addWidget(title)

        posture_subtabs = QTabWidget()
        postures = [
            ('姿勢1', '例：0°', 'posture1'),
            ('姿勢2', '例：45°', 'posture2'),
            ('姿勢3', '例：90°', 'posture3'),
        ]
        posture_widgets = {}

        for posture_label, example_text, posture_key in postures:
            posture_widget = self._create_posture_view_widget(
                posture_label, example_text, posture_key, axis_letter
            )
            posture_widgets[posture_label] = posture_widget
            posture_subtabs.addTab(posture_widget, posture_label)

        # 姿勢3 の次に「X軸回転軸 / 並進軸」タブを追加
        # 姿勢ごとの表示切替コントロールは今は U axis のみ有効化（要望どおり段階的に導入）
        enable_posture_ctrls = (axis_letter == 'u')
        motion_widget = self._create_motion_axis_tab(
            axis_letter, posture_widgets, joint_type=joint_type,
            enable_posture_controls=enable_posture_ctrls,
        )
        tab_label = f'{axis_letter.upper()}軸回転軸' if joint_type == 'rotation' else f'{axis_letter.upper()}軸並進軸'
        posture_subtabs.addTab(motion_widget, tab_label)

        main_layout.addWidget(posture_subtabs, 1)

        self.axis_data[axis_letter] = {
            'subtabs': posture_subtabs,
            'posture_widgets': posture_widgets,
            'motion_widget': motion_widget,
            'joint_type': joint_type,
        }

        return widget

    def _create_motion_axis_tab(self, axis_letter: str, posture_widgets: dict, joint_type: str = 'rotation', enable_posture_controls: bool = False) -> QWidget:
        """姿勢1/2/3 を C_world で重ね、{axis_letter}軸の motion 軸を求めて表示するタブ。
        joint_type='rotation' → U/V/W (回転軸を計算)
        joint_type='translation' → X/Y/Z (並進軸を計算)
        """
        widget = QWidget()
        widget.axis_letter = axis_letter
        widget.posture_widgets = posture_widgets
        widget.joint_type = joint_type
        if joint_type == 'rotation':
            widget.motion_label_jp = '回転軸'
            widget.motion_suffix_en = 'rotation-axis'
        else:
            widget.motion_label_jp = '並進軸'
            widget.motion_suffix_en = 'parallel-axis'
        widget.motion_name = f'{axis_letter.upper()}_{widget.motion_suffix_en}'

        main_layout = QVBoxLayout(widget)

        top_layout = QHBoxLayout()

        # === 左パネル: 操作 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel(f'{axis_letter.upper()}軸 {widget.motion_label_jp}の計算（C_world 上で重ね合わせ）')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        left_layout.addWidget(title)

        compute_btn = QPushButton(f'{widget.motion_label_jp}を計算 / 表示更新')
        left_layout.addWidget(compute_btn)

        opacity_group = QGroupBox('STL 透明度')
        opacity_layout = QVBoxLayout(opacity_group)

        sliders = {}
        for posture_label, default_val in (('姿勢1', 50), ('姿勢2', 50), ('姿勢3', 50)):
            row = QHBoxLayout()
            lbl = QLabel(f'{posture_label}:')
            lbl.setMinimumWidth(48)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(default_val)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(default_val)
            spin.setSuffix('%')
            spin.setMinimumWidth(70)
            # 双方向同期
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            opacity_layout.addLayout(row)
            sliders[posture_label] = (slider, spin)

        left_layout.addWidget(opacity_group)

        # === 軸の表示切替 ===
        axis_vis_group = QGroupBox(f'{widget.motion_name} の表示')
        axis_vis_layout = QHBoxLayout(axis_vis_group)
        show_line_cb = QCheckBox('軸全体')
        show_line_cb.setChecked(True)
        show_arrow_cb = QCheckBox('矢印')
        show_arrow_cb.setChecked(True)
        axis_vis_layout.addWidget(show_line_cb)
        axis_vis_layout.addWidget(show_arrow_cb)
        axis_vis_layout.addStretch()
        left_layout.addWidget(axis_vis_group)
        widget.show_line_cb = show_line_cb
        widget.show_arrow_cb = show_arrow_cb
        widget.show_axis_line = True
        widget.show_axis_arrow = True
        show_line_cb.toggled.connect(
            lambda checked, w=widget: self._on_motion_axis_visibility_toggled(w, 'line', checked)
        )
        show_arrow_cb.toggled.connect(
            lambda checked, w=widget: self._on_motion_axis_visibility_toggled(w, 'arrow', checked)
        )

        # === 姿勢変化の表示切替（U axis のみ有効化）===
        # デフォルト状態（無効化時も同じデフォルトを使用 → 表示挙動は従来と同じ）
        widget.unify_stl_color = False
        widget.stl_visibility = {f'姿勢{i}': True for i in (1, 2, 3)}
        widget.caxis_visibility = {f'姿勢{i}': True for i in (1, 2, 3)}

        if enable_posture_controls:
            posture_view_group = QGroupBox('姿勢変化を分かりやすくする')
            pv_layout = QVBoxLayout(posture_view_group)

            unify_cb = QCheckBox('色を統一')
            unify_cb.setChecked(False)
            pv_layout.addWidget(unify_cb)

            stl_cbs = {}
            caxis_cbs = {}
            for i in (1, 2, 3):
                p_label = f'姿勢{i}'
                caxis_label = f'C_{axis_letter}-axis_posi{i}'
                row = QHBoxLayout()
                stl_cb = QCheckBox(f'{p_label} を表示')
                stl_cb.setChecked(True)
                cax_cb = QCheckBox(f'{caxis_label} を表示')
                cax_cb.setChecked(True)
                row.addWidget(stl_cb)
                row.addWidget(cax_cb)
                pv_layout.addLayout(row)
                stl_cbs[p_label] = stl_cb
                caxis_cbs[p_label] = cax_cb

            left_layout.addWidget(posture_view_group)

            widget.unify_color_cb = unify_cb
            widget.stl_visibility_cbs = stl_cbs
            widget.caxis_visibility_cbs = caxis_cbs

            unify_cb.toggled.connect(
                lambda c, w=widget: self._on_posture_view_toggled(w, 'unify', None, c)
            )
            for p, cb in stl_cbs.items():
                cb.toggled.connect(
                    lambda c, w=widget, pl=p: self._on_posture_view_toggled(w, 'stl', pl, c)
                )
            for p, cb in caxis_cbs.items():
                cb.toggled.connect(
                    lambda c, w=widget, pl=p: self._on_posture_view_toggled(w, 'caxis', pl, c)
                )

        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(280)
        left_scroll.setMaximumWidth(420)

        # === 右パネル: 3D ビュー ===
        if HAS_PYVISTA:
            plotter = QtInteractor(widget)
            background_color, _ = self._load_visual_settings()
            plotter.set_background(background_color, top=self._background_top_color(background_color))
            plotter.add_text(f'「{widget.motion_label_jp}を計算」ボタンを押してください', position='upper_left', font_size=10)
            self._configure_lights(plotter)
            right_view = plotter.interactor
        else:
            plotter = None
            right_view = QLabel('pyvista / pyvistaqt が未インストール')
            right_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top_layout.addWidget(left_scroll, 1)
        top_layout.addWidget(right_view, 4)

        main_layout.addLayout(top_layout, 5)

        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setPlaceholderText('ログ')
        log_view.setMinimumHeight(140)
        log_view.setMaximumHeight(260)
        main_layout.addWidget(log_view, 1)

        widget.plotter = plotter
        widget.log_view = log_view
        widget.compute_btn = compute_btn
        widget.sliders = sliders
        widget.motion_axis = None
        widget.bounds_all = None
        self.visual_widgets.append(widget)

        compute_btn.clicked.connect(lambda: self._compute_motion_axis(widget))
        for posture_label, (slider, _spin) in sliders.items():
            slider.valueChanged.connect(
                lambda v, pl=posture_label, w=widget: self._on_rotation_opacity_changed(w, pl, v)
            )

        return widget

    def _on_motion_axis_visibility_toggled(self, widget, kind, checked):
        if kind == 'line':
            widget.show_axis_line = bool(checked)
        elif kind == 'arrow':
            widget.show_axis_arrow = bool(checked)
        self._render_motion_view(widget)

    def _on_posture_view_toggled(self, widget, kind, posture_label, checked):
        if kind == 'unify':
            widget.unify_stl_color = bool(checked)
        elif kind == 'stl':
            if not hasattr(widget, 'stl_visibility'):
                widget.stl_visibility = {}
            widget.stl_visibility[posture_label] = bool(checked)
        elif kind == 'caxis':
            if not hasattr(widget, 'caxis_visibility'):
                widget.caxis_visibility = {}
            widget.caxis_visibility[posture_label] = bool(checked)
        self._render_motion_view(widget)

    def _on_rotation_opacity_changed(self, widget, posture_label, value):
        # QSpinBox は slider.valueChanged ↔ spin.valueChanged で自動同期するため、ここでは表示更新不要
        plotter = getattr(widget, 'plotter', None)
        if plotter is None:
            return
        actor_name = f'rot_stl_{posture_label}'
        try:
            actor = plotter.renderer.actors.get(actor_name)
        except Exception:
            actor = None
        if actor is not None:
            try:
                actor.GetProperty().SetOpacity(value / 100.0)
                plotter.render()
            except Exception:
                pass

    @staticmethod
    def _rotation_log_axis(R):
        """3x3 回転行列の log（軸角表現）を返す。
        Returns: (theta_radians [0, pi], axis_unit_vector (3,))
        """
        R = np.asarray(R, dtype=float)
        cos_theta = (np.trace(R) - 1.0) / 2.0
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = float(np.arccos(cos_theta))
        if theta < 1e-9:
            return 0.0, np.array([1.0, 0.0, 0.0])
        if abs(theta - np.pi) < 1e-6:
            # 180°: 軸は (R + I) の列空間（対称成分から取り出す）
            M = (R + np.eye(3)) / 2.0
            i = int(np.argmax(np.diag(M)))
            axis = M[:, i]
            n = np.linalg.norm(axis)
            if n < 1e-9:
                return theta, np.array([1.0, 0.0, 0.0])
            return theta, axis / n
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * np.sin(theta))
        return theta, axis

    def _compute_motion_axis(self, widget):
        log = widget.log_view
        log.setText('')

        axis_letter = getattr(widget, 'axis_letter', 'u')
        joint_type = getattr(widget, 'joint_type', 'rotation')
        motion_name = getattr(widget, 'motion_name', f'{axis_letter.upper()}_motion-axis')
        local_prefix = f'C_{axis_letter}-axis'
        posture_widgets = getattr(widget, 'posture_widgets', {})
        postures = ['姿勢1', '姿勢2', '姿勢3']

        # 入力検証
        for p in postures:
            pw = posture_widgets.get(p)
            if pw is None:
                log.append(f'{p}: ウィジェットが見つかりません。')
                return
            if getattr(pw, 'c_axis', None) is None:
                log.append(f'{p}: {local_prefix} 座標系が未生成です。各姿勢で先に生成してください。')
                return
            if getattr(pw, 'c_world', None) is None:
                log.append(f'{p}: C_world 座標系が未生成です。各姿勢で先に生成してください。')
                return
            if getattr(pw, 'current_mesh', None) is None:
                log.append(f'{p}: STL が読み込まれていません。')
                return

        # 各姿勢: STL→World 変換、および C_local の World 表現
        R_local_world = []
        O_local_world = []
        T_stl_to_world = []
        for p in postures:
            pw = posture_widgets[p]
            R_loc = np.column_stack([pw.c_axis['ex'], pw.c_axis['ey'], pw.c_axis['ez']])
            O_loc = np.asarray(pw.c_axis['origin'], dtype=float)
            R_w = np.column_stack([pw.c_world['ex'], pw.c_world['ey'], pw.c_world['ez']])
            O_w = np.asarray(pw.c_world['origin'], dtype=float)

            R_w_T = R_w.T
            R_local_world.append(R_w_T @ R_loc)
            O_local_world.append(R_w_T @ (O_loc - O_w))

            T = np.eye(4)
            T[:3, :3] = R_w_T
            T[:3, 3] = -R_w_T @ O_w
            T_stl_to_world.append(T)

        # 姿勢間の回転（World 座標系上）
        R_12 = R_local_world[1] @ R_local_world[0].T
        R_23 = R_local_world[2] @ R_local_world[1].T
        R_13 = R_local_world[2] @ R_local_world[0].T

        theta_12, axis_12 = self._rotation_log_axis(R_12)
        theta_23, axis_23 = self._rotation_log_axis(R_23)
        theta_13, axis_13 = self._rotation_log_axis(R_13)

        # 姿勢間の平行移動（World 座標系上）
        d_12 = O_local_world[1] - O_local_world[0]
        d_23 = O_local_world[2] - O_local_world[1]
        d_13 = O_local_world[2] - O_local_world[0]

        def _deg(v, w):
            nv = np.linalg.norm(v); nw = np.linalg.norm(w)
            if nv < 1e-12 or nw < 1e-12:
                return float('nan')
            return float(np.degrees(np.arccos(np.clip(float(np.dot(v / nv, w / nw)), -1.0, 1.0))))

        if joint_type == 'rotation':
            # 軸方向の符号合わせ
            if np.dot(axis_12, axis_23) < 0:
                axis_23 = -axis_23
            if np.dot(axis_12, axis_13) < 0:
                axis_13 = -axis_13

            weights = np.array([theta_12, theta_23, theta_13])
            if weights.sum() < 1e-9:
                log.append('回転角がほぼゼロです（3 姿勢が同一）。回転軸を確定できません。')
                return
            avg_dir = weights[0] * axis_12 + weights[1] * axis_23 + weights[2] * axis_13
            avg_dir = avg_dir / np.linalg.norm(avg_dir)

            # 回転軸の通る点 p を最小二乗で
            I3 = np.eye(3)
            A = np.vstack([I3 - R_12, I3 - R_23, I3 - R_13])
            b = np.concatenate([
                O_local_world[1] - R_12 @ O_local_world[0],
                O_local_world[2] - R_23 @ O_local_world[1],
                O_local_world[2] - R_13 @ O_local_world[0],
            ])
            try:
                p_axis, _r, _rk, _sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                log.append('回転軸の位置を解けませんでした。')
                return
            p_axis = p_axis - np.dot(p_axis, avg_dir) * avg_dir

            widget.motion_axis = {
                'direction': avg_dir,
                'point': p_axis,
                'R_local_world': R_local_world,
                'O_local_world': O_local_world,
                'T_stl_to_world': T_stl_to_world,
            }

            log.append(f'{motion_name} を計算しました（C_world 座標系上）:')
            log.append(f'  姿勢1→2: 回転角 = {np.degrees(theta_12):.3f}°, 軸 = ({axis_12[0]:+.4f}, {axis_12[1]:+.4f}, {axis_12[2]:+.4f})')
            log.append(f'  姿勢2→3: 回転角 = {np.degrees(theta_23):.3f}°, 軸 = ({axis_23[0]:+.4f}, {axis_23[1]:+.4f}, {axis_23[2]:+.4f})')
            log.append(f'  姿勢1→3: 回転角 = {np.degrees(theta_13):.3f}°, 軸 = ({axis_13[0]:+.4f}, {axis_13[1]:+.4f}, {axis_13[2]:+.4f})')
            log.append('  --- 統合結果 ---')
            log.append(f'  {motion_name} 方向 = ({avg_dir[0]:+.6f}, {avg_dir[1]:+.6f}, {avg_dir[2]:+.6f})')
            log.append(f'  軸が通る点 (C_world)   = ({p_axis[0]:+.4f}, {p_axis[1]:+.4f}, {p_axis[2]:+.4f})')
            log.append(
                f'  3軸候補の方向ずれ: '
                f'∠(1→2, 2→3)={_deg(axis_12, axis_23):.3f}°, '
                f'∠(2→3, 1→3)={_deg(axis_23, axis_13):.3f}°, '
                f'∠(1→2, 1→3)={_deg(axis_12, axis_13):.3f}°（理想 0°）'
            )
        else:  # translation
            # 並進方向の符号合わせ
            d_23_s = d_23 if np.dot(d_12, d_23) >= 0 else -d_23
            d_13_s = d_13 if np.dot(d_12, d_13) >= 0 else -d_13

            lens = np.array([np.linalg.norm(d_12), np.linalg.norm(d_23_s), np.linalg.norm(d_13_s)])
            if lens.sum() < 1e-9:
                log.append('並進距離がほぼゼロです（3 姿勢が同一原点）。並進軸を確定できません。')
                return
            avg_dir = lens[0] * d_12 + lens[1] * d_23_s + lens[2] * d_13_s
            avg_dir = avg_dir / np.linalg.norm(avg_dir)

            # 軸の通る点: 姿勢1 の C_local 原点を採用
            p_axis = O_local_world[0]

            widget.motion_axis = {
                'direction': avg_dir,
                'point': p_axis,
                'R_local_world': R_local_world,
                'O_local_world': O_local_world,
                'T_stl_to_world': T_stl_to_world,
            }

            log.append(f'{motion_name} を計算しました（C_world 座標系上）:')
            log.append('  ※ 姿勢間の回転行列 R（直動関節なら理想 R = I）と平行移動 Δ を表示します。')

            def _fmt_R(R):
                return [
                    f'    [{R[0, 0]:+.6f}, {R[0, 1]:+.6f}, {R[0, 2]:+.6f}]',
                    f'    [{R[1, 0]:+.6f}, {R[1, 1]:+.6f}, {R[1, 2]:+.6f}]',
                    f'    [{R[2, 0]:+.6f}, {R[2, 1]:+.6f}, {R[2, 2]:+.6f}]',
                ]
            for label, R, theta, dvec in (
                ('1→2', R_12, theta_12, d_12),
                ('2→3', R_23, theta_23, d_23),
                ('1→3', R_13, theta_13, d_13),
            ):
                log.append(f'  --- 姿勢{label} ---')
                log.append(f'    R_{label.replace("→", "_")} (回転角 = {np.degrees(theta):.4f}°, 理想 0°):')
                for ln in _fmt_R(R):
                    log.append(ln)
                log.append(
                    f'    Δ_{label.replace("→", "_")} = '
                    f'({dvec[0]:+.4f}, {dvec[1]:+.4f}, {dvec[2]:+.4f}) [mm], '
                    f'||Δ|| = {np.linalg.norm(dvec):.4f} mm'
                )

            log.append('  --- 統合結果 ---')
            log.append(f'  {motion_name} 方向 = ({avg_dir[0]:+.6f}, {avg_dir[1]:+.6f}, {avg_dir[2]:+.6f})')
            log.append(f'  軸が通る点 (C_world, 姿勢1 原点) = ({p_axis[0]:+.4f}, {p_axis[1]:+.4f}, {p_axis[2]:+.4f})')
            log.append(
                f'  3 候補方向のずれ: '
                f'∠(Δ_12, Δ_23) = {_deg(d_12, d_23_s):.4f}°, '
                f'∠(Δ_23, Δ_13) = {_deg(d_23_s, d_13_s):.4f}°, '
                f'∠(Δ_12, Δ_13) = {_deg(d_12, d_13_s):.4f}° (理想 0°)'
            )

        self._render_motion_view(widget)

    def _render_motion_view(self, widget):
        plotter = getattr(widget, 'plotter', None)
        if plotter is None or not HAS_PYVISTA:
            return

        plotter.clear()
        background_color, model_color = self._load_visual_settings()
        plotter.set_background(background_color, top=self._background_top_color(background_color))
        self._configure_lights(plotter)
        plotter.hide_axes()

        rot = getattr(widget, 'motion_axis', None)
        if rot is None:
            plotter.render()
            return

        axis_letter = getattr(widget, 'axis_letter', 'u')
        motion_name = getattr(widget, 'motion_name', f'{axis_letter.upper()}_motion-axis')
        local_label_prefix = f'C_{axis_letter}-axis'
        posture_widgets = getattr(widget, 'posture_widgets', {})

        T_list = rot['T_stl_to_world']
        postures = ['姿勢1', '姿勢2', '姿勢3']
        # 姿勢ごとに異なる淡い色で重ねる
        stl_colors = {
            '姿勢1': '#ffc070',
            '姿勢2': '#80d0a0',
            '姿勢3': '#a0a0ff',
        }
        stl_visibility = getattr(widget, 'stl_visibility', {})
        unify_color = getattr(widget, 'unify_stl_color', False)
        _bg, unified_model_color = self._load_visual_settings()

        bounds_all = None

        def _accumulate_bounds(bnds, b):
            if bnds is None:
                return list(b)
            return [
                min(bnds[0], b[0]), max(bnds[1], b[1]),
                min(bnds[2], b[2]), max(bnds[3], b[3]),
                min(bnds[4], b[4]), max(bnds[5], b[5]),
            ]

        for p, T in zip(postures, T_list):
            pw = posture_widgets[p]
            mesh = pw.current_mesh.copy()
            mesh.transform(T, inplace=True)
            b = mesh.bounds
            # 非表示でも bounds は更新（カメラ安定化のため）
            bounds_all = _accumulate_bounds(bounds_all, b)

            if not stl_visibility.get(p, True):
                continue

            opacity = widget.sliders[p][0].value() / 100.0
            color = unified_model_color if unify_color else stl_colors[p]
            try:
                plotter.add_mesh(
                    mesh,
                    name=f'rot_stl_{p}',
                    color=color,
                    opacity=opacity,
                    show_edges=False,
                    smooth_shading=True,
                    ambient=0.15,
                    diffuse=0.75,
                    specular=0.35,
                    specular_power=25.0,
                    pickable=False,
                )
            except Exception:
                continue
        widget.bounds_all = bounds_all

        if bounds_all is None:
            plotter.render()
            return

        diag = float(np.linalg.norm([
            bounds_all[1] - bounds_all[0],
            bounds_all[3] - bounds_all[2],
            bounds_all[5] - bounds_all[4],
        ])) or 100.0

        # C_world の座標軸（原点 = 0, 単位ベクトル）
        axis_len_world = max(diag * 0.18, 1.0)
        for name, vec, color in (
            ('cw_x', np.array([1.0, 0.0, 0.0]), '#ff3030'),
            ('cw_y', np.array([0.0, 1.0, 0.0]), '#3060ff'),
            ('cw_z', np.array([0.0, 0.0, 1.0]), '#30c030'),
        ):
            try:
                arrow = pv.Arrow(start=np.zeros(3), direction=vec, scale=axis_len_world,
                                 shaft_radius=0.012, tip_radius=0.04, tip_length=0.18)
                plotter.add_mesh(arrow, name=name, color=color, pickable=False, reset_camera=False, render=False)
            except Exception:
                pass
        # C_world ラベル
        try:
            offs = axis_len_world * 0.12
            plotter.add_point_labels(
                np.array([[offs, offs, offs]], dtype=float),
                ['C_world'],
                name='cw_label',
                font_size=14,
                text_color='#bfe4ff',
                point_size=0,
                shape=None,
                always_visible=True,
                pickable=False,
                reset_camera=False,
                render=False,
            )
        except Exception:
            pass

        # 各姿勢の C_local-axis を C_world 上で薄く表示（参考）
        caxis_visibility = getattr(widget, 'caxis_visibility', {})
        axis_len_local = max(diag * 0.12, 1.0)
        for i, p in enumerate(postures, 1):
            if not caxis_visibility.get(p, True):
                continue
            R_lw = rot['R_local_world'][i - 1]
            O_lw = rot['O_local_world'][i - 1]
            for k, (col_idx, color) in enumerate((
                (0, '#ff8080'),  # X 軸
                (1, '#8080ff'),  # Y 軸
                (2, '#80ff80'),  # Z 軸
            )):
                try:
                    arrow = pv.Arrow(
                        start=O_lw, direction=R_lw[:, col_idx], scale=axis_len_local,
                        shaft_radius=0.010, tip_radius=0.035, tip_length=0.18,
                    )
                    plotter.add_mesh(arrow, name=f'cloc_{i}_{k}', color=color, opacity=0.85,
                                     pickable=False, reset_camera=False, render=False)
                except Exception:
                    pass
            try:
                offs = axis_len_local * 0.15
                label_pos = O_lw + np.array([offs, offs, offs])
                plotter.add_point_labels(
                    np.array([label_pos], dtype=float),
                    [f'{local_label_prefix}_posi{i}'],
                    name=f'cloc_label_{i}',
                    font_size=12,
                    text_color='#ffe0a0',
                    point_size=0,
                    shape=None,
                    always_visible=True,
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

        # 回転軸 / 並進軸（黄色の太い直線 + 矢印 + ラベル）
        show_line = getattr(widget, 'show_axis_line', True)
        show_arrow = getattr(widget, 'show_axis_arrow', True)
        direction = rot['direction']
        point = rot['point']
        line_half = diag * 0.9
        start_pt = point - direction * line_half
        end_pt = point + direction * line_half
        if show_line:
            try:
                line = pv.Line(start_pt, end_pt)
                plotter.add_mesh(line, name='rot_axis_line', color='#ffd000',
                                 line_width=4, pickable=False, reset_camera=False, render=False)
            except Exception:
                pass
        if show_line and show_arrow:
            try:
                arrow = pv.Arrow(
                    start=point, direction=direction, scale=diag * 0.35,
                    shaft_radius=0.018, tip_radius=0.055, tip_length=0.20,
                )
                plotter.add_mesh(arrow, name='rot_axis_arrow', color='#ffd000',
                                 pickable=False, reset_camera=False, render=False)
            except Exception:
                pass
        if show_line:
            try:
                label_pos = point + direction * (diag * 0.4)
                label_pos = label_pos + np.array([diag * 0.03, diag * 0.03, diag * 0.03])
                plotter.add_point_labels(
                    np.array([label_pos], dtype=float),
                    [motion_name],
                    name='rot_axis_label',
                    font_size=16,
                    text_color='#ffe060',
                    point_size=0,
                    shape=None,
                    always_visible=True,
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

        plotter.reset_camera(bounds=bounds_all)
        plotter.render()

    def _create_plane_point_controls_widget(self, plane_label: str, plane_title: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel(plane_title)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 11px;')
        layout.addWidget(title)

        self._add_plane_point_controls(widget, layout)

        layout.addStretch()

        widget.plane_label = plane_label
        widget.current_mesh = None
        widget.points = []
        widget.selected_point_index = -1
        widget.point_add_enabled = False
        return widget

    def _create_posture_view_widget(self, posture_label: str, example_text: str, posture_key: str, axis_letter: str = 'u') -> QWidget:
        widget = DnDWidget()
        widget.posture_key = posture_key
        widget.axis_letter = axis_letter
        # 例: 'C_u-axis_posi1', 'C_v-axis_posi1', 'C_w-axis_posi1'
        widget.c_axis_name = f'C_{axis_letter}-axis_{posture_key.replace("posture", "posi")}'
        widget.c_axis_label_prefix = f'C_{axis_letter}-axis'  # 例 'C_u-axis'
        main_layout = QVBoxLayout(widget)

        # Ver.1 風レイアウト: 上段 = 左コントロール + 右 3D ビュー、下段 = ログ
        top_layout = QHBoxLayout()

        # === 左パネル: タイトル / 読み込み / 軸 / 平面サブタブ ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel(f'{posture_label}  {example_text}')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        left_layout.addWidget(title)

        load_btn = QPushButton('STLを読み込む')
        left_layout.addWidget(load_btn)

        build_axis_btn = QPushButton(f'{widget.c_axis_label_prefix} 座標系を生成')
        build_axis_btn.setEnabled(False)
        left_layout.addWidget(build_axis_btn)

        clear_axis_btn = QPushButton(f'{widget.c_axis_label_prefix} 座標系を消去')
        clear_axis_btn.setEnabled(False)
        left_layout.addWidget(clear_axis_btn)

        build_world_btn = QPushButton('C_world 座標系を生成')
        build_world_btn.setEnabled(False)
        left_layout.addWidget(build_world_btn)

        clear_world_btn = QPushButton('C_world 座標系を消去')
        clear_world_btn.setEnabled(False)
        left_layout.addWidget(clear_world_btn)

        # === 右パネル: 共有 3D ビュー ===
        if HAS_PYVISTA:
            plotter = QtInteractor(widget)
            background_color, _ = self._load_visual_settings()
            plotter.set_background(background_color, top=self._background_top_color(background_color))
            plotter.add_text('STLを読み込んでください', position='upper_left', font_size=10)
            self._configure_lights(plotter)
            right_view = plotter.interactor
        else:
            plotter = None
            right_view = QLabel('pyvista / pyvistaqt が未インストール')
            right_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        widget.plotter = plotter

        # === 下段: ログビュー（全幅） ===
        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setPlaceholderText('ログ')
        log_view.setMinimumHeight(120)
        log_view.setMaximumHeight(220)

        # 平面サブタブ（点コントロールのみ。3D ビューは共有）
        # 前半 3 タブが C_u-axis 用、後半 3 タブが C_world 用
        plane_subtabs = QTabWidget()
        widget.plane_subtabs = plane_subtabs
        cax = widget.c_axis_label_prefix  # 'C_u-axis' / 'C_v-axis' / 'C_w-axis'
        plane_specs = [
            # (plane_label, plane_title, system_type)
            ('平面1（XY平面）', f'XY平面 [{cax} 用]', 'c_axis'),
            ('平面2（YZ平面）', f'YZ平面 [{cax} 用]', 'c_axis'),
            ('平面3（ZX平面）', f'ZX平面 [{cax} 用]', 'c_axis'),
            ('W平面1（XY平面）', 'W平面1（XY平面） [C_world 用]', 'c_world'),
            ('W平面2（YZ平面）', 'W平面2（YZ平面） [C_world 用]', 'c_world'),
            ('W平面3（ZX平面）', 'W平面3（ZX平面） [C_world 用]', 'c_world'),
        ]

        plane_widgets = []
        widget.shared_points = {}        # C_u-axis 用
        widget.shared_world_points = {}  # C_world 用
        widget.active_plane_index = 0

        for plane_label, plane_title, system_type in plane_specs:
            plane_widget = self._create_plane_point_controls_widget(plane_label, plane_title)
            plane_widget.posture_widget = widget
            plane_widget.plotter = plotter  # 共有プロッタを参照
            plane_widget.log_view = log_view  # 共有ログ
            plane_widget.system_type = system_type
            target_dict = widget.shared_points if system_type == 'c_axis' else widget.shared_world_points
            plane_widget.points = target_dict.setdefault(plane_label, [])
            self._wire_plane_point_handlers(plane_widget)
            plane_subtabs.addTab(plane_widget, plane_label)
            plane_widgets.append(plane_widget)

        left_layout.addWidget(plane_subtabs, 1)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(280)
        left_scroll.setMaximumWidth(420)

        top_layout.addWidget(left_scroll, 1)
        top_layout.addWidget(right_view, 4)

        main_layout.addLayout(top_layout, 5)
        main_layout.addWidget(log_view, 1)

        widget.load_btn = load_btn
        widget.build_axis_btn = build_axis_btn
        widget.clear_axis_btn = clear_axis_btn
        widget.build_world_btn = build_world_btn
        widget.clear_world_btn = clear_world_btn
        widget.log_view = log_view
        widget.plane_widgets = plane_widgets
        widget.current_mesh = None
        widget.c_axis = None
        widget.c_world = None
        self.visual_widgets.append(widget)

        def _start_load(path: str, preserve_state: bool = False):
            widget._next_load_preserve = preserve_state
            widget._pending_load_path = path
            widget.log_view.setText('')
            widget.log_view.append(
                f'読込要求: {path}' + ('（キャッシュ復元）' if preserve_state else '')
            )
            widget.load_btn.setEnabled(False)

            widget.thread = QThread(widget)
            widget.worker = STLLoadWorker(path)
            widget.worker.moveToThread(widget.thread)

            widget.thread.started.connect(widget.worker.run)
            widget.worker.log.connect(lambda msg: widget.log_view.append(msg))
            widget.worker.finished.connect(lambda mesh: _on_mesh_loaded(mesh))
            widget.worker.error.connect(lambda msg: _on_load_error(msg))

            widget.worker.finished.connect(widget.thread.quit)
            widget.worker.error.connect(widget.thread.quit)
            widget.worker.finished.connect(widget.worker.deleteLater)
            widget.worker.error.connect(widget.worker.deleteLater)
            widget.thread.finished.connect(widget.thread.deleteLater)

            widget.thread.start()

        def _on_mesh_loaded(mesh):
            preserve = bool(getattr(widget, '_next_load_preserve', False))
            widget._next_load_preserve = False
            widget.current_mesh = mesh
            widget.stl_path = getattr(widget, '_pending_load_path', None) or getattr(widget, 'stl_path', None)
            widget.log_view.append('表示を更新します...')

            if not preserve:
                widget.c_axis = None
                widget.c_world = None
                for plane_widget in widget.plane_widgets:
                    plane_widget.current_mesh = mesh
                    plane_widget.points.clear()
                    plane_widget.selected_point_index = -1
                    plane_widget.point_add_enabled = False
                    plane_widget.point_add_btn.setChecked(False)
                    plane_widget.point_add_btn.setEnabled(True)
                    plane_widget._refresh_point_list()
            else:
                # キャッシュから復元した点群・座標系を保ち、メッシュだけ差し替える
                for plane_widget in widget.plane_widgets:
                    plane_widget.current_mesh = mesh
                    plane_widget.point_add_enabled = False
                    plane_widget.point_add_btn.setChecked(False)
                    plane_widget.point_add_btn.setEnabled(True)
                    if plane_widget.points:
                        plane_widget.selected_point_index = len(plane_widget.points) - 1
                    else:
                        plane_widget.selected_point_index = -1
                    plane_widget._refresh_point_list()

            self._render_posture1_plotter(widget, reset_view=True)
            widget.build_axis_btn.setEnabled(True)
            widget.clear_axis_btn.setEnabled(widget.c_axis is not None)
            widget.build_world_btn.setEnabled(True)
            widget.clear_world_btn.setEnabled(widget.c_world is not None)
            widget.log_view.append('完了')
            widget.load_btn.setEnabled(True)
            self._save_posture_cache(widget)

        def _on_load_error(msg: str):
            widget.log_view.append(msg)
            widget.load_btn.setEnabled(True)

        load_btn.clicked.connect(lambda: self._open_posture_file(widget))
        build_axis_btn.clicked.connect(lambda: self._build_c_axis(widget))
        clear_axis_btn.clicked.connect(lambda: self._clear_c_axis(widget))
        build_world_btn.clicked.connect(lambda: self._build_c_world_axis(widget))
        clear_world_btn.clicked.connect(lambda: self._clear_c_world_axis(widget))

        widget._start_load = _start_load
        widget._on_mesh_loaded = _on_mesh_loaded
        widget._on_load_error = _on_load_error

        # STL ファイルの D&D を受け付ける
        if isinstance(widget, DnDWidget):
            widget.set_dnd_callback(lambda p: widget._start_load(p))

        def _refresh_all_plane_views(reset_view: bool = False):
            # 共有プロッタを 1 度だけ再描画
            self._render_posture1_plotter(widget, reset_view=reset_view)

        widget._refresh_all_plane_views = _refresh_all_plane_views

        def _on_plane_subtab_changed(index: int):
            widget.active_plane_index = index
            if index < 0 or index >= len(widget.plane_widgets):
                return
            if widget.plotter is None or widget.current_mesh is None:
                return
            # ピッカーをアクティブな平面のコールバックへ切り替え
            active = widget.plane_widgets[index]
            try:
                widget.plotter.disable_picking()
            except Exception:
                pass
            if active.point_add_enabled:
                try:
                    widget.plotter.enable_surface_point_picking(
                        callback=lambda point, *_args: self._on_plane_surface_point_picked(active, point),
                        left_clicking=True,
                        show_point=False,
                        pickable_window=False,
                    )
                except Exception:
                    pass
            # アクティブ平面の選択ハイライトを反映するため再描画（カメラは維持）
            self._render_posture1_plotter(widget, reset_view=False)

        widget._on_plane_subtab_changed = _on_plane_subtab_changed
        plane_subtabs.currentChanged.connect(_on_plane_subtab_changed)

        # キャッシュ（点群・C_u-axis・STLパス）を復元。STL があれば自動読込。
        widget.stl_path = None
        cached_stl_path = self._load_posture_cache(widget)
        for plane_widget in plane_widgets:
            plane_widget._refresh_point_list()
        if cached_stl_path and os.path.exists(cached_stl_path):
            widget.log_view.append(f'キャッシュ検出: {cached_stl_path}')
            widget._start_load(cached_stl_path, preserve_state=True)
        elif cached_stl_path:
            widget.log_view.append(f'前回のSTLが見つかりません: {cached_stl_path}')

        return widget

    def _add_plane_point_controls(self, widget, left_layout):
        point_add_btn = QPushButton('点追加モード: OFF')
        point_add_btn.setCheckable(True)
        point_add_btn.setEnabled(False)
        left_layout.addWidget(point_add_btn)

        point_list = QListWidget()
        point_list.setMinimumHeight(120)
        point_list.setMaximumHeight(220)
        left_layout.addWidget(point_list)

        delete_selected_btn = QPushButton('選択点を削除')
        delete_selected_btn.setEnabled(False)
        delete_last_btn = QPushButton('最後の点を削除')
        delete_last_btn.setEnabled(False)
        clear_points_btn = QPushButton('全削除')
        clear_points_btn.setEnabled(False)
        left_layout.addWidget(delete_selected_btn)
        left_layout.addWidget(delete_last_btn)
        left_layout.addWidget(clear_points_btn)

        widget.point_add_btn = point_add_btn
        widget.point_list = point_list
        widget.delete_selected_btn = delete_selected_btn
        widget.delete_last_btn = delete_last_btn
        widget.clear_points_btn = clear_points_btn

    def _wire_plane_point_handlers(self, widget):
        def _refresh_point_list():
            widget.point_list.blockSignals(True)
            widget.point_list.clear()
            for idx, point in enumerate(getattr(widget, 'points', [])):
                widget.point_list.addItem(f'{idx + 1}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})')
            if 0 <= getattr(widget, 'selected_point_index', -1) < len(getattr(widget, 'points', [])):
                widget.point_list.setCurrentRow(widget.selected_point_index)
            else:
                widget.selected_point_index = -1
            widget.point_list.blockSignals(False)
            self._update_plane_point_buttons(widget)

        def _update_selected_point_actor():
            if widget.plotter is None:
                return
            try:
                widget.plotter.remove_actor('selected_point')
            except Exception:
                pass
            points = getattr(widget, 'points', [])
            if 0 <= widget.selected_point_index < len(points):
                selected = np.array([points[widget.selected_point_index]], dtype=float)
                widget.plotter.add_mesh(
                    pv.PolyData(selected),
                    name='selected_point',
                    color='#ffff66',
                    point_size=18,
                    render_points_as_spheres=True,
                    style='points',
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
            widget.plotter.render()

        def _on_point_list_selection_changed(row: int):
            widget.selected_point_index = row
            self._update_plane_point_buttons(widget)
            _update_selected_point_actor()

        def _on_point_add_toggled(checked: bool):
            if checked and widget.current_mesh is None:
                widget.log_view.append('先に姿勢1でSTLを読み込んでください。')
                widget.point_add_btn.setChecked(False)
                return
            widget.point_add_btn.setText('点追加モード: ON' if checked else '点追加モード: OFF')
            widget.point_add_enabled = checked
            if widget.plotter is not None:
                try:
                    if checked:
                        widget.plotter.enable_surface_point_picking(
                            callback=lambda point, *_args: self._on_plane_surface_point_picked(widget, point),
                            left_clicking=True,
                            show_point=False,
                            pickable_window=False,
                        )
                    else:
                        widget.plotter.disable_picking()
                except Exception:
                    pass

        def _on_delete_selected():
            points = getattr(widget, 'points', [])
            if 0 <= widget.selected_point_index < len(points):
                del points[widget.selected_point_index]
                widget.selected_point_index = min(widget.selected_point_index, len(points) - 1)
                _refresh_point_list()
                widget.posture_widget._refresh_all_plane_views(reset_view=False)
                self._save_posture_cache(widget.posture_widget)

        def _on_delete_last():
            points = getattr(widget, 'points', [])
            if points:
                points.pop()
                widget.selected_point_index = min(widget.selected_point_index, len(points) - 1)
                _refresh_point_list()
                widget.posture_widget._refresh_all_plane_views(reset_view=False)
                self._save_posture_cache(widget.posture_widget)

        def _on_clear_points():
            widget.points.clear()
            widget.selected_point_index = -1
            _refresh_point_list()
            widget.posture_widget._refresh_all_plane_views(reset_view=False)
            self._save_posture_cache(widget.posture_widget)

        def _on_point_picked(row: int):
            widget.selected_point_index = row
            _refresh_point_list()

        widget._refresh_point_list = _refresh_point_list
        widget._update_selected_point_actor = _update_selected_point_actor
        widget._on_point_list_selection_changed = _on_point_list_selection_changed
        widget._on_point_add_toggled = _on_point_add_toggled
        widget._on_delete_selected = _on_delete_selected
        widget._on_delete_last = _on_delete_last
        widget._on_clear_points = _on_clear_points
        widget._on_point_picked = _on_point_picked

        widget.point_list.currentRowChanged.connect(widget._on_point_list_selection_changed)
        widget.point_add_btn.toggled.connect(widget._on_point_add_toggled)
        widget.delete_selected_btn.clicked.connect(widget._on_delete_selected)
        widget.delete_last_btn.clicked.connect(widget._on_delete_last)
        widget.clear_points_btn.clicked.connect(widget._on_clear_points)

    def _update_plane_point_buttons(self, widget):
        points = getattr(widget, 'points', [])
        has_points = len(points) > 0
        has_selected = 0 <= getattr(widget, 'selected_point_index', -1) < len(points)
        widget.delete_selected_btn.setEnabled(has_selected)
        widget.delete_last_btn.setEnabled(has_points)
        widget.clear_points_btn.setEnabled(has_points)

    def _on_plane_surface_point_picked(self, widget, point, *_args):
        if not getattr(widget, 'point_add_enabled', False):
            return
        if point is None:
            return

        if getattr(widget, 'points', None) is None:
            widget.points = []

        p = np.array(point, dtype=float)
        widget.points.append(p)
        widget.selected_point_index = len(widget.points) - 1
        widget.log_view.append(
            f'点追加 [{widget.plane_label}] #{widget.selected_point_index + 1}: '
            f'({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})'
        )
        widget._refresh_point_list()
        widget.posture_widget._refresh_all_plane_views(reset_view=False)
        self._save_posture_cache(widget.posture_widget)

    def _render_posture1_plotter(self, posture_widget, reset_view: bool = False):
        plotter = getattr(posture_widget, 'plotter', None)
        if plotter is None or posture_widget.current_mesh is None:
            return

        try:
            plotter.disable_picking()
        except Exception:
            pass

        camera_position = None
        if not reset_view:
            try:
                camera_position = plotter.camera_position
            except Exception:
                camera_position = None

        plotter.clear()
        background_color, model_color = self._load_visual_settings()
        plotter.set_background(background_color, top=self._background_top_color(background_color))
        self._configure_lights(plotter)
        plotter.add_mesh(
            posture_widget.current_mesh,
            name='stl_model',
            color=model_color,
            show_edges=False,
            smooth_shading=True,
            ambient=0.15,
            diffuse=0.75,
            specular=0.35,
            specular_power=25.0,
            pickable=True,
        )
        plotter.hide_axes()

        plane_colors = {
            '平面1（XY平面）': '#ff0000',
            '平面2（YZ平面）': '#0000ff',
            '平面3（ZX平面）': '#00ff00',
            'W平面1（XY平面）': '#ff8c00',
            'W平面2（YZ平面）': '#00d4d4',
            'W平面3（ZX平面）': '#d040d0',
        }
        for points_dict in (posture_widget.shared_points, getattr(posture_widget, 'shared_world_points', {})):
            for plane_label, points in points_dict.items():
                if not points:
                    continue
                points_array = np.array(points, dtype=float)
                plotter.add_mesh(
                    pv.PolyData(points_array),
                    name=f'plane_points::{plane_label}',
                    color=plane_colors.get(plane_label, '#ffffff'),
                    point_size=12,
                    render_points_as_spheres=True,
                    style='points',
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )

        active_index = getattr(posture_widget, 'active_plane_index', 0)
        active_plane = None
        if 0 <= active_index < len(posture_widget.plane_widgets):
            active_plane = posture_widget.plane_widgets[active_index]
            pts = getattr(active_plane, 'points', [])
            sel = getattr(active_plane, 'selected_point_index', -1)
            if 0 <= sel < len(pts):
                selected = np.array([pts[sel]], dtype=float)
                plotter.add_mesh(
                    pv.PolyData(selected),
                    name='selected_point',
                    color='#ffff66',
                    point_size=18,
                    render_points_as_spheres=True,
                    style='points',
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )

        # Build surface planes from points (match Tab1 behavior)
        surface_plane_colors = {
            '平面1（XY平面）': '#ff0000',
            '平面2（YZ平面）': '#0000ff',
            '平面3（ZX平面）': '#00ff00',
            'W平面1（XY平面）': '#ff8c00',
            'W平面2（YZ平面）': '#00d4d4',
            'W平面3（ZX平面）': '#d040d0',
        }
        for points_dict in (posture_widget.shared_points, getattr(posture_widget, 'shared_world_points', {})):
            for plane_label, points in points_dict.items():
                if not points or len(points) < 3:
                    continue
                try:
                    plane = self._build_plane_from_points(points)
                except Exception:
                    plane = None
                if plane is None:
                    continue
                is_selected = (getattr(active_plane, 'plane_label', None) == plane_label)
                try:
                    plotter.add_mesh(
                        plane,
                        name=f'plane_surface::{plane_label}',
                        color=surface_plane_colors.get(plane_label, '#ffffff'),
                        opacity=0.45 if is_selected else 0.20,
                        pickable=False,
                        reset_camera=False,
                        render=False,
                        show_edges=False,
                        smooth_shading=True,
                    )
                except Exception:
                    pass

        self._draw_c_axis(posture_widget)

        if reset_view:
            plotter.reset_camera(bounds=posture_widget.current_mesh.bounds)
        elif camera_position is not None:
            try:
                plotter.camera_position = camera_position
            except Exception:
                pass

        if active_plane is not None and active_plane.point_add_enabled:
            try:
                plotter.enable_surface_point_picking(
                    callback=lambda point, *_args: self._on_plane_surface_point_picked(active_plane, point),
                    left_clicking=True,
                    show_point=False,
                    pickable_window=False,
                )
            except Exception:
                pass

        for pw in posture_widget.plane_widgets:
            if hasattr(pw, 'point_add_btn'):
                pw.point_add_btn.setEnabled(pw.current_mesh is not None)

        plotter.render()

    def _fit_plane_basis(self, points):
        arr = np.array(points, dtype=float)
        if arr.shape[0] < 3:
            return None

        center = np.mean(arr, axis=0)
        centered = arr - center
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except Exception:
            return None

        normal = vh[-1]
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-9:
            return None
        normal = normal / n_norm

        u_vec = vh[0]
        u_norm = np.linalg.norm(u_vec)
        if u_norm < 1e-9:
            return None
        u_vec = u_vec / u_norm
        v_vec = np.cross(normal, u_vec)
        v_norm = np.linalg.norm(v_vec)
        if v_norm < 1e-9:
            return None
        v_vec = v_vec / v_norm

        return center, normal, u_vec, v_vec

    def _build_plane_from_points(self, points):
        fit = self._fit_plane_basis(points)
        if fit is None:
            return None
        center, normal, u_vec, v_vec = fit

        arr = np.array(points, dtype=float)
        centered = arr - center
        x = np.dot(centered, u_vec)
        y = np.dot(centered, v_vec)
        spread = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0))
        plane_size = max(spread * 2.4, 1.0)

        return pv.Plane(
            center=center,
            direction=normal,
            i_size=plane_size,
            j_size=plane_size,
            i_resolution=1,
            j_resolution=1,
        )

    def _serialize_points_dict(self, points_dict):
        return {
            plane_label: [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
            for plane_label, pts in points_dict.items()
        }

    def _restore_points_dict(self, points_dict, cached):
        for plane_label, plane_pts in points_dict.items():
            plane_pts.clear()
            for raw in (cached.get(plane_label) or []):
                if isinstance(raw, (list, tuple)) and len(raw) == 3:
                    try:
                        plane_pts.append(np.array(raw, dtype=float))
                    except Exception:
                        continue

    def _serialize_frame(self, frame):
        if frame is None:
            return None
        return {
            key: np.asarray(frame[key], dtype=float).tolist()
            for key in ('origin', 'ex', 'ey', 'ez', 'raw_x', 'raw_y', 'raw_z')
        }

    def _deserialize_frame(self, raw):
        if not isinstance(raw, dict):
            return None
        try:
            return {
                key: np.array(raw[key], dtype=float)
                for key in ('origin', 'ex', 'ey', 'ez', 'raw_x', 'raw_y', 'raw_z')
            }
        except Exception:
            return None

    def _save_posture_cache(self, posture_widget):
        """姿勢ごとの点群・C_[u/v/w]-axis・C_world・STL パスを保存する。"""
        posture_key = getattr(posture_widget, 'posture_key', None)
        axis_letter = getattr(posture_widget, 'axis_letter', 'u')
        if not posture_key:
            return
        try:
            settings = load_settings() or {}
            tab2 = settings.setdefault('tab2', {})
            axis_section = tab2.setdefault(f'{axis_letter}_axis', {})
            posture_entry = axis_section.setdefault(posture_key, {})

            stl_path = getattr(posture_widget, 'stl_path', None)
            if stl_path:
                posture_entry['stl_path'] = stl_path
            else:
                posture_entry.pop('stl_path', None)

            posture_entry['points'] = self._serialize_points_dict(posture_widget.shared_points)
            posture_entry['world_points'] = self._serialize_points_dict(
                getattr(posture_widget, 'shared_world_points', {})
            )

            c_local = self._serialize_frame(getattr(posture_widget, 'c_axis', None))
            if c_local is not None:
                posture_entry['c_axis'] = c_local
            else:
                posture_entry.pop('c_axis', None)
            # 旧キー（U 軸）を削除して整理
            posture_entry.pop('c_u_axis', None)

            c_world = self._serialize_frame(getattr(posture_widget, 'c_world', None))
            if c_world is not None:
                posture_entry['c_world'] = c_world
            else:
                posture_entry.pop('c_world', None)

            save_settings(settings)
        except Exception:
            pass

    def _load_posture_cache(self, posture_widget):
        """設定から姿勢の状態を取り出す。点群と座標系はその場で復元する。

        STL パス（あれば）は呼び出し側で `_start_load(path, preserve_state=True)` する。
        """
        posture_key = getattr(posture_widget, 'posture_key', None)
        axis_letter = getattr(posture_widget, 'axis_letter', 'u')
        if not posture_key:
            return ''
        try:
            settings = load_settings() or {}
            posture_entry = (
                ((settings.get('tab2') or {}).get(f'{axis_letter}_axis') or {}).get(posture_key) or {}
            )
        except Exception:
            return ''

        self._restore_points_dict(posture_widget.shared_points, posture_entry.get('points') or {})
        if hasattr(posture_widget, 'shared_world_points'):
            cached_world = dict(posture_entry.get('world_points') or {})
            # 旧キー（W平面1/2/3）→ 新キー（W平面1（XY平面）等）への移行
            migration_map = {
                'W平面1': 'W平面1（XY平面）',
                'W平面2': 'W平面2（YZ平面）',
                'W平面3': 'W平面3（ZX平面）',
            }
            for old_key, new_key in migration_map.items():
                if old_key in cached_world and new_key not in cached_world:
                    cached_world[new_key] = cached_world.pop(old_key)
            self._restore_points_dict(posture_widget.shared_world_points, cached_world)

        # 旧キー 'c_u_axis' も読み込んで後方互換（U 軸の既存キャッシュ用）
        posture_widget.c_axis = self._deserialize_frame(
            posture_entry.get('c_axis') or posture_entry.get('c_u_axis')
        )
        posture_widget.c_world = self._deserialize_frame(posture_entry.get('c_world'))

        return str(posture_entry.get('stl_path') or '')

    def _compute_axis_system(self, plane_points_in_order, log, prefix: str):
        """3 平面の点群から、直交化済み座標系の dict を返す（共通ロジック）。

        plane_points_in_order: 3 つの (N,3) numpy 配列のリスト（plane1, plane2, plane3 の順）
        log: ログ表示用の QTextEdit
        prefix: ログ／エラーメッセージ用のプレフィックス（例 'C_u-axis', 'C_world'）
        戻り値: dict (origin, ex, ey, ez, raw_x, raw_y, raw_z) または None
        """
        for i, arr in enumerate(plane_points_in_order, 1):
            if arr.shape[0] < 3:
                log.append(f'{prefix}: 平面{i}には3点以上が必要です（現在 {arr.shape[0]} 点）。')
                return None

        centroids = []
        normals = []
        for i, arr in enumerate(plane_points_in_order, 1):
            fit = self._fit_plane_basis(arr)
            if fit is None:
                log.append(f'{prefix}: 平面{i}のフィッティングに失敗しました。')
                return None
            c, n, _u, _v = fit
            centroids.append(c)
            normals.append(np.array(n, dtype=float))

        # 法線の符号正規化：各 n_i を「他 2 平面の点群の重心」へ向ける
        for i in range(3):
            others = np.vstack([plane_points_in_order[j] for j in range(3) if j != i])
            ref_dir = others.mean(axis=0) - centroids[i]
            if np.dot(normals[i], ref_dir) < 0:
                normals[i] = -normals[i]

        n1, n2, n3 = normals
        c1, c2, c3 = centroids

        # 原点：n_i・x = n_i・c_i の連立解
        A = np.vstack([n1, n2, n3])
        d = np.array([np.dot(n1, c1), np.dot(n2, c2), np.dot(n3, c3)])
        try:
            origin = np.linalg.solve(A, d)
        except np.linalg.LinAlgError:
            log.append(f'{prefix}: 3 平面の交点が一意に決まりません（平面が平行または一次従属です）。')
            return None

        def _safe_normalize(v, name):
            nn = np.linalg.norm(v)
            if nn < 1e-9:
                log.append(f'{prefix}: {name} を構築できません（平面の法線が平行）。')
                return None
            return v / nn

        raw_x = _safe_normalize(np.cross(n3, n1), 'X 軸')   # 平面1 ∩ 平面3
        raw_y = _safe_normalize(np.cross(n1, n2), 'Y 軸')   # 平面1 ∩ 平面2
        raw_z = _safe_normalize(np.cross(n2, n3), 'Z 軸')   # 平面2 ∩ 平面3
        if raw_x is None or raw_y is None or raw_z is None:
            return None

        # 最近接の正規直交行列（極分解 / Procrustes）
        M = np.column_stack([raw_x, raw_y, raw_z])
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ Vt
        e_x = R[:, 0]
        e_y = R[:, 1]
        e_z = R[:, 2]

        # X 軸と Z 軸を反転（Y はそのまま、右手系保存）
        e_x = -e_x
        e_z = -e_z
        raw_x = -raw_x
        raw_z = -raw_z

        def _deg(v, w):
            return float(np.degrees(np.arccos(np.clip(float(np.dot(v, w)), -1.0, 1.0))))

        log.append(f'{prefix} 座標系を構築しました:')
        log.append(f'  原点 O = ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})')
        log.append('  [直交化前: 生の平面交線]')
        log.append(f'    X_raw(平面1∩平面3) = ({raw_x[0]:+.4f}, {raw_x[1]:+.4f}, {raw_x[2]:+.4f})')
        log.append(f'    Y_raw(平面1∩平面2) = ({raw_y[0]:+.4f}, {raw_y[1]:+.4f}, {raw_y[2]:+.4f})')
        log.append(f'    Z_raw(平面2∩平面3) = ({raw_z[0]:+.4f}, {raw_z[1]:+.4f}, {raw_z[2]:+.4f})')
        log.append(
            f'    角度（理想 90°）: '
            f'∠(X_raw, Y_raw) = {_deg(raw_x, raw_y):.4f}°, '
            f'∠(Y_raw, Z_raw) = {_deg(raw_y, raw_z):.4f}°, '
            f'∠(Z_raw, X_raw) = {_deg(raw_z, raw_x):.4f}°'
        )
        log.append(f'  [直交化後: {prefix} 基底（表示ベクトル）]')
        log.append(f'    X(赤) e_x = ({e_x[0]:+.4f}, {e_x[1]:+.4f}, {e_x[2]:+.4f})')
        log.append(f'    Y(青) e_y = ({e_y[0]:+.4f}, {e_y[1]:+.4f}, {e_y[2]:+.4f})')
        log.append(f'    Z(緑) e_z = ({e_z[0]:+.4f}, {e_z[1]:+.4f}, {e_z[2]:+.4f})')
        log.append(
            f'    角度（厳密 90°）: '
            f'∠(e_x, e_y) = {_deg(e_x, e_y):.4f}°, '
            f'∠(e_y, e_z) = {_deg(e_y, e_z):.4f}°, '
            f'∠(e_z, e_x) = {_deg(e_z, e_x):.4f}°'
        )

        return {
            'origin': origin,
            'ex': e_x, 'ey': e_y, 'ez': e_z,
            'raw_x': raw_x, 'raw_y': raw_y, 'raw_z': raw_z,
        }

    def _collect_plane_points(self, points_dict, plane_keys):
        """点群 dict から指定キー順に numpy 配列リストを取り出す。"""
        return [
            np.array(list(points_dict.get(k, [])), dtype=float) if points_dict.get(k) else np.zeros((0, 3))
            for k in plane_keys
        ]

    def _build_c_axis(self, posture_widget):
        plane_keys = ['平面1（XY平面）', '平面2（YZ平面）', '平面3（ZX平面）']
        pts = self._collect_plane_points(posture_widget.shared_points, plane_keys)
        prefix = getattr(posture_widget, 'c_axis_label_prefix', 'C_u-axis')
        result = self._compute_axis_system(pts, posture_widget.log_view, prefix=prefix)
        if result is None:
            return
        posture_widget.c_axis = result
        posture_widget.clear_axis_btn.setEnabled(True)
        self._render_posture1_plotter(posture_widget, reset_view=False)
        self._save_posture_cache(posture_widget)

    def _build_c_world_axis(self, posture_widget):
        plane_keys = ['W平面1（XY平面）', 'W平面2（YZ平面）', 'W平面3（ZX平面）']
        pts = self._collect_plane_points(posture_widget.shared_world_points, plane_keys)
        result = self._compute_axis_system(pts, posture_widget.log_view, prefix='C_world')
        if result is None:
            return
        posture_widget.c_world = result
        posture_widget.clear_world_btn.setEnabled(True)
        self._render_posture1_plotter(posture_widget, reset_view=False)
        self._save_posture_cache(posture_widget)

    def _clear_c_axis(self, posture_widget):
        if getattr(posture_widget, 'c_axis', None) is None:
            return
        posture_widget.c_axis = None
        posture_widget.clear_axis_btn.setEnabled(False)
        prefix = getattr(posture_widget, 'c_axis_label_prefix', 'C_u-axis')
        posture_widget.log_view.append(f'{prefix} 座標系を消去しました。')
        self._render_posture1_plotter(posture_widget, reset_view=False)
        self._save_posture_cache(posture_widget)

    def _clear_c_world_axis(self, posture_widget):
        if getattr(posture_widget, 'c_world', None) is None:
            return
        posture_widget.c_world = None
        posture_widget.clear_world_btn.setEnabled(False)
        posture_widget.log_view.append('C_world 座標系を消去しました。')
        self._render_posture1_plotter(posture_widget, reset_view=False)
        self._save_posture_cache(posture_widget)

    def _draw_c_axis(self, posture_widget):
        """姿勢上の 2 つの座標系（C_u-axis_posi_i と C_world）を 3D ビューに描画。"""
        plotter = getattr(posture_widget, 'plotter', None)
        if plotter is None or not HAS_PYVISTA:
            return

        # スケール基準
        mesh = posture_widget.current_mesh
        if mesh is not None:
            b = mesh.bounds
            diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
        else:
            diag = 100.0
        axis_len = max(diag * 0.2, 1.0)

        systems = [
            {
                'key': 'c_axis',
                'frame': getattr(posture_widget, 'c_axis', None),
                'label': getattr(posture_widget, 'c_axis_name', 'C_u-axis'),
                'label_color': '#ffffaa',
                'origin_color': '#ffff66',
                'arrow_colors': ('#ff3030', '#3060ff', '#30c030'),  # X red / Y blue / Z green
                'actor_prefix': 'c_axis',
            },
            {
                'key': 'c_world',
                'frame': getattr(posture_widget, 'c_world', None),
                'label': 'C_world',
                'label_color': '#bfe4ff',
                'origin_color': '#80c0ff',
                # 同色だと混乱するが、ユーザー指定の規約（X=赤 / Y=青 / Z=緑）に揃える
                'arrow_colors': ('#ff3030', '#3060ff', '#30c030'),
                'actor_prefix': 'c_world',
            },
        ]

        for sys_def in systems:
            prefix = sys_def['actor_prefix']
            # 既存アクター除去
            for suffix in ('_x', '_y', '_z', '_origin', '_label'):
                try:
                    plotter.remove_actor(prefix + suffix)
                except Exception:
                    pass

            frame = sys_def['frame']
            if frame is None:
                continue

            origin = frame['origin']
            cx, cy, cz = sys_def['arrow_colors']
            for suffix, vec, color in (
                ('_x', frame['ex'], cx),
                ('_y', frame['ey'], cy),
                ('_z', frame['ez'], cz),
            ):
                try:
                    arrow = pv.Arrow(
                        start=origin,
                        direction=vec,
                        scale=axis_len,
                        shaft_radius=0.015,
                        tip_radius=0.045,
                        tip_length=0.20,
                    )
                    plotter.add_mesh(
                        arrow,
                        name=prefix + suffix,
                        color=color,
                        pickable=False,
                        reset_camera=False,
                        render=False,
                    )
                except Exception:
                    pass

            try:
                plotter.add_mesh(
                    pv.PolyData(np.array([origin], dtype=float)),
                    name=prefix + '_origin',
                    color=sys_def['origin_color'],
                    point_size=14,
                    render_points_as_spheres=True,
                    style='points',
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

            # 座標系名のラベル（原点から軸長 10% オフセット）
            label_offset = axis_len * 0.10
            label_pos = np.array(origin, dtype=float) + np.array(
                [label_offset, label_offset, label_offset], dtype=float
            )
            try:
                plotter.add_point_labels(
                    np.array([label_pos], dtype=float),
                    [sys_def['label']],
                    name=prefix + '_label',
                    font_size=14,
                    text_color=sys_def['label_color'],
                    point_size=0,
                    shape=None,
                    always_visible=True,
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

    def _open_posture_file(self, widget):
        path, _ = QFileDialog.getOpenFileName(widget, 'STLファイルを開く', '', 'STL Files (*.stl)')
        if not path:
            return
        widget._start_load(path)
