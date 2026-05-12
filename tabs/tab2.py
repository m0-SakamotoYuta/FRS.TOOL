from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QGroupBox, QRadioButton,
    QButtonGroup, QScrollArea, QPushButton, QFileDialog, QTextEdit, QListWidget
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
        self.posture_widgets = {}
        self.visual_widgets = []
        layout = QVBoxLayout(self)
        
        # サブタブウィジェットを作成
        subtabs = QTabWidget()
        
        # 各軸のサブタブ
        axis_names = ['ALL VIEW', 'U axis', 'V axis', 'W axis', 'X axis', 'Y axis', 'Z axis']
        for axis_name in axis_names:
            if axis_name == 'U axis':
                axis_widget = self._create_u_axis_tab()
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
    
    def _create_u_axis_tab(self) -> QWidget:
        """U axis タブを作成。姿勢ごとに独立した STL/点群/C_u-axis を持つ。"""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)

        title = QLabel('U axis')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        main_layout.addWidget(title)

        self.posture_subtabs = QTabWidget()
        postures = [
            ('姿勢1', '例：0°', 'posture1'),
            ('姿勢2', '例：45°', 'posture2'),
            ('姿勢3', '例：90°', 'posture3'),
        ]
        self.posture_widgets = {}

        for posture_label, example_text, posture_key in postures:
            posture_widget = self._create_posture_view_widget(posture_label, example_text, posture_key)
            self.posture_widgets[posture_label] = posture_widget
            self.posture_subtabs.addTab(posture_widget, posture_label)

        main_layout.addWidget(self.posture_subtabs, 1)

        return widget

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

    def _create_posture_view_widget(self, posture_label: str, example_text: str, posture_key: str) -> QWidget:
        widget = QWidget()
        widget.posture_key = posture_key
        # 'posture1' → 'C_u-axis_posi1' のように座標系名を組み立てる
        widget.c_u_axis_name = f'C_u-axis_{posture_key.replace("posture", "posi")}'
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

        build_axis_btn = QPushButton('C_u-axis 座標系を生成')
        build_axis_btn.setEnabled(False)
        left_layout.addWidget(build_axis_btn)

        clear_axis_btn = QPushButton('C_u-axis 座標系を消去')
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
        plane_specs = [
            # (plane_label, plane_title, system_type)
            ('平面1（XY平面）', 'XY平面 [C_u-axis 用]', 'c_u_axis'),
            ('平面2（YZ平面）', 'YZ平面 [C_u-axis 用]', 'c_u_axis'),
            ('平面3（ZX平面）', 'ZX平面 [C_u-axis 用]', 'c_u_axis'),
            ('W平面1', 'W平面1 [C_world 用]', 'c_world'),
            ('W平面2', 'W平面2 [C_world 用]', 'c_world'),
            ('W平面3', 'W平面3 [C_world 用]', 'c_world'),
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
            target_dict = widget.shared_points if system_type == 'c_u_axis' else widget.shared_world_points
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
        widget.c_u_axis = None
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
                widget.c_u_axis = None
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
            widget.clear_axis_btn.setEnabled(widget.c_u_axis is not None)
            widget.build_world_btn.setEnabled(True)
            widget.clear_world_btn.setEnabled(widget.c_world is not None)
            widget.log_view.append('完了')
            widget.load_btn.setEnabled(True)
            self._save_posture_cache(widget)

        def _on_load_error(msg: str):
            widget.log_view.append(msg)
            widget.load_btn.setEnabled(True)

        load_btn.clicked.connect(lambda: self._open_posture_file(widget))
        build_axis_btn.clicked.connect(lambda: self._build_c_u_axis(widget))
        clear_axis_btn.clicked.connect(lambda: self._clear_c_u_axis(widget))
        build_world_btn.clicked.connect(lambda: self._build_c_world_axis(widget))
        clear_world_btn.clicked.connect(lambda: self._clear_c_world_axis(widget))

        widget._start_load = _start_load
        widget._on_mesh_loaded = _on_mesh_loaded
        widget._on_load_error = _on_load_error

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
            'W平面1': '#ff8c00',
            'W平面2': '#00d4d4',
            'W平面3': '#d040d0',
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
            'W平面1': '#ff8c00',
            'W平面2': '#00d4d4',
            'W平面3': '#d040d0',
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

        self._draw_c_u_axis(posture_widget)

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
        """姿勢ごとの点群・C_u-axis・C_world・STL パスをユーザー設定に保存する。"""
        posture_key = getattr(posture_widget, 'posture_key', None)
        if not posture_key:
            return
        try:
            settings = load_settings() or {}
            tab2 = settings.setdefault('tab2', {})
            u_axis = tab2.setdefault('u_axis', {})
            posture_entry = u_axis.setdefault(posture_key, {})

            stl_path = getattr(posture_widget, 'stl_path', None)
            if stl_path:
                posture_entry['stl_path'] = stl_path
            else:
                posture_entry.pop('stl_path', None)

            posture_entry['points'] = self._serialize_points_dict(posture_widget.shared_points)
            posture_entry['world_points'] = self._serialize_points_dict(
                getattr(posture_widget, 'shared_world_points', {})
            )

            c_u = self._serialize_frame(getattr(posture_widget, 'c_u_axis', None))
            if c_u is not None:
                posture_entry['c_u_axis'] = c_u
            else:
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
        if not posture_key:
            return ''
        try:
            settings = load_settings() or {}
            posture_entry = (
                ((settings.get('tab2') or {}).get('u_axis') or {}).get(posture_key) or {}
            )
        except Exception:
            return ''

        self._restore_points_dict(posture_widget.shared_points, posture_entry.get('points') or {})
        if hasattr(posture_widget, 'shared_world_points'):
            self._restore_points_dict(
                posture_widget.shared_world_points, posture_entry.get('world_points') or {}
            )

        posture_widget.c_u_axis = self._deserialize_frame(posture_entry.get('c_u_axis'))
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

    def _build_c_u_axis(self, posture_widget):
        plane_keys = ['平面1（XY平面）', '平面2（YZ平面）', '平面3（ZX平面）']
        pts = self._collect_plane_points(posture_widget.shared_points, plane_keys)
        result = self._compute_axis_system(pts, posture_widget.log_view, prefix='C_u-axis')
        if result is None:
            return
        posture_widget.c_u_axis = result
        posture_widget.clear_axis_btn.setEnabled(True)
        self._render_posture1_plotter(posture_widget, reset_view=False)
        self._save_posture_cache(posture_widget)

    def _build_c_world_axis(self, posture_widget):
        plane_keys = ['W平面1', 'W平面2', 'W平面3']
        pts = self._collect_plane_points(posture_widget.shared_world_points, plane_keys)
        result = self._compute_axis_system(pts, posture_widget.log_view, prefix='C_world')
        if result is None:
            return
        posture_widget.c_world = result
        posture_widget.clear_world_btn.setEnabled(True)
        self._render_posture1_plotter(posture_widget, reset_view=False)
        self._save_posture_cache(posture_widget)

    def _clear_c_u_axis(self, posture_widget):
        if getattr(posture_widget, 'c_u_axis', None) is None:
            return
        posture_widget.c_u_axis = None
        posture_widget.clear_axis_btn.setEnabled(False)
        posture_widget.log_view.append('C_u-axis 座標系を消去しました。')
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

    def _draw_c_u_axis(self, posture_widget):
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
                'key': 'c_u_axis',
                'frame': getattr(posture_widget, 'c_u_axis', None),
                'label': getattr(posture_widget, 'c_u_axis_name', 'C_u-axis'),
                'label_color': '#ffffaa',
                'origin_color': '#ffff66',
                'arrow_colors': ('#ff3030', '#3060ff', '#30c030'),  # X red / Y blue / Z green
                'actor_prefix': 'c_u_axis',
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
