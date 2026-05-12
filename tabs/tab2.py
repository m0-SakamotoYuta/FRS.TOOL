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

from tabs.settings import load_settings


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
        """U axis タブを作成。姿勢ラジオボタンと、各姿勢の STL読み込み + 3D表示。"""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        
        # U axis タイトル
        title = QLabel('U axis')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        main_layout.addWidget(title)
        
        # 姿勢ごとの STL ロード + 3D表示エリア（タブウィジェット）
        self.posture_subtabs = QTabWidget()
        postures = [
            ('姿勢1', '例：0°'),
            ('姿勢2', '例：45°'),
            ('姿勢3', '例：90°'),
        ]
        self.posture_widgets = {}
        
        for posture_label, example_text in postures:
            if posture_label == '姿勢1':
                posture_widget = self._create_posture1_view_widget(posture_label, example_text)
            else:
                posture_widget = self._create_posture_view_widget(posture_label, example_text)
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

    def _create_posture1_view_widget(self, posture_label: str, example_text: str) -> QWidget:
        widget = QWidget()
        main_layout = QVBoxLayout(widget)

        title = QLabel(f'{posture_label}  {example_text}')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        main_layout.addWidget(title)

        load_btn = QPushButton('STLを読み込む')
        main_layout.addWidget(load_btn)

        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setPlaceholderText('ログ')
        log_view.setMinimumHeight(80)
        log_view.setMaximumHeight(140)
        main_layout.addWidget(log_view)

        # 単一の共有 3D ビュー（タブ切替で変化しない）
        if HAS_PYVISTA:
            plotter = QtInteractor(widget)
            background_color, _ = self._load_visual_settings()
            plotter.set_background(background_color, top=self._background_top_color(background_color))
            plotter.add_text('STLを読み込んでください', position='upper_left', font_size=10)
            self._configure_lights(plotter)
            main_layout.addWidget(plotter.interactor, 3)
        else:
            plotter = None
            fallback = QLabel('pyvista / pyvistaqt が未インストール')
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(fallback, 3)

        widget.plotter = plotter

        # 平面サブタブ（点コントロールのみ。3D ビューは共有）
        plane_subtabs = QTabWidget()
        widget.plane_subtabs = plane_subtabs
        plane_specs = [
            ('平面1（XY平面）', 'XY平面'),
            ('平面2（YZ平面）', 'YZ平面'),
            ('平面3（ZX平面）', 'ZX平面'),
        ]

        plane_widgets = []
        widget.shared_points = {}
        widget.active_plane_index = 0

        for plane_label, plane_title in plane_specs:
            plane_widget = self._create_plane_point_controls_widget(plane_label, plane_title)
            plane_widget.posture_widget = widget
            plane_widget.plotter = plotter  # 共有プロッタを参照
            plane_widget.log_view = log_view  # 共有ログ
            plane_widget.points = widget.shared_points.setdefault(plane_label, [])
            self._wire_plane_point_handlers(plane_widget)
            plane_subtabs.addTab(plane_widget, plane_label)
            plane_widgets.append(plane_widget)

        main_layout.addWidget(plane_subtabs, 2)

        widget.load_btn = load_btn
        widget.log_view = log_view
        widget.plane_widgets = plane_widgets
        widget.current_mesh = None
        self.visual_widgets.append(widget)

        def _start_load(path: str):
            widget.log_view.setText('')
            widget.log_view.append(f'読込要求: {path}')
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
            widget.current_mesh = mesh
            widget.log_view.append('表示を更新します...')

            for plane_widget in widget.plane_widgets:
                plane_widget.current_mesh = mesh
                plane_widget.points.clear()
                plane_widget.selected_point_index = -1
                plane_widget.point_add_enabled = False
                plane_widget.point_add_btn.setChecked(False)
                plane_widget.point_add_btn.setEnabled(True)
                plane_widget._refresh_point_list()

            self._render_posture1_plotter(widget, reset_view=True)
            widget.log_view.append('完了')
            widget.load_btn.setEnabled(True)

        def _on_load_error(msg: str):
            widget.log_view.append(msg)
            widget.load_btn.setEnabled(True)

        load_btn.clicked.connect(lambda: self._open_posture1_file(widget))

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

        def _on_delete_last():
            points = getattr(widget, 'points', [])
            if points:
                points.pop()
                widget.selected_point_index = min(widget.selected_point_index, len(points) - 1)
                _refresh_point_list()
                widget.posture_widget._refresh_all_plane_views(reset_view=False)

        def _on_clear_points():
            widget.points.clear()
            widget.selected_point_index = -1
            _refresh_point_list()
            widget.posture_widget._refresh_all_plane_views(reset_view=False)

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
        }
        for plane_label, points in posture_widget.shared_points.items():
            if not points:
                continue
            points_array = np.array(points, dtype=float)
            plotter.add_mesh(
                pv.PolyData(points_array),
                name=f'plane_points::{plane_label}',
                color=plane_colors.get(plane_label, '#ff0000'),
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
        plane_colors = {
            '平面1（XY平面）': '#ff0000',
            '平面2（YZ平面）': '#0000ff',
            '平面3（ZX平面）': '#00ff00',
        }
        for plane_label, points in posture_widget.shared_points.items():
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
                    pickable=False,
                    reset_camera=False,
                    render=False,
                    show_edges=False,
                    smooth_shading=True,
                )
            except Exception:
                pass

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

    def _open_posture1_file(self, widget):
        path, _ = QFileDialog.getOpenFileName(widget, 'STLファイルを開く', '', 'STL Files (*.stl)')
        if not path:
            return
        widget._start_load(path)
    
    def _create_posture_view_widget(self, posture_label: str, example_text: str) -> QWidget:
        """各姿勢用の STL読み込み + 3D表示エリアを作成。"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # 左側：ボタンとログ
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        load_btn = QPushButton(f'{posture_label}: STLを読み込む')
        left_layout.addWidget(load_btn)
        
        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setPlaceholderText('ログ')
        log_view.setMinimumHeight(100)
        left_layout.addWidget(log_view)
        
        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(300)
        
        # 右側：3D表示
        if HAS_PYVISTA:
            plotter = QtInteractor(widget)
            background_color, _ = self._load_visual_settings()
            plotter.set_background(background_color, top=self._background_top_color(background_color))
            plotter.add_text('STLを読み込んでください', position='upper_left', font_size=10)
            self._configure_lights(plotter)
            right_panel = plotter.interactor
        else:
            right_panel = QLabel('pyvista / pyvistaqt が未インストール')
            right_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(left_panel, 0)
        layout.addWidget(right_panel, 1)
        
        # データ保存用属性
        widget.posture_label = posture_label
        widget.load_btn = load_btn
        widget.log_view = log_view
        widget.plotter = plotter if HAS_PYVISTA else None
        widget.current_mesh = None
        
        # ボタンクリック時の処理
        def on_load_click():
            path, _ = QFileDialog.getOpenFileName(widget, 'STLファイルを開く', '', 'STL Files (*.stl)')
            if not path:
                return
            widget._start_load(path)
        
        load_btn.clicked.connect(on_load_click)
        
        # ロード処理メソッド
        def _start_load(path: str):
            widget.log_view.setText('')
            widget.log_view.append(f'読込要求: {path}')
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
            widget.current_mesh = mesh
            widget.log_view.append('表示を更新します...')
            
            if widget.plotter is not None:
                try:
                    widget.plotter.disable_picking()
                except Exception:
                    pass
                widget.plotter.clear()
                background_color, model_color = self._load_visual_settings()
                widget.plotter.set_background(background_color, top=self._background_top_color(background_color))
                self._configure_lights(widget.plotter)
                widget.plotter.add_mesh(
                    mesh,
                    name='stl_model',
                    color=model_color,
                    show_edges=False,
                    smooth_shading=True,
                    ambient=0.15,
                    diffuse=0.75,
                    specular=0.35,
                    specular_power=25.0,
                )
                widget.plotter.hide_axes()
                widget.plotter.reset_camera(bounds=mesh.bounds)
                widget.plotter.render()
                widget.log_view.append('完了')
            
            widget.load_btn.setEnabled(True)
        
        def _on_load_error(msg: str):
            widget.log_view.append(msg)
            widget.load_btn.setEnabled(True)
        
        widget._start_load = _start_load
        widget._on_mesh_loaded = _on_mesh_loaded
        widget._on_load_error = _on_load_error
        
        return widget
