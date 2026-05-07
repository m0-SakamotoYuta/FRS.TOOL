import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTextEdit,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
    QListWidget,
    QColorDialog,
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, QEvent
from PyQt6.QtGui import QColor

from tabs.settings import load_settings, save_settings, export_points_data, import_points_data
from splash import LoadingDialog

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False


class STLLoadWorker(QObject):
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
            # 立体感を出すため表示用法線を作成
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


class Tab1Widget(QWidget):
    """0点校正タブ: STL読込とインタラクティブ表示まで（点打ち・円柱生成は未実装）。"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_path = ''
        self.current_mesh = None
        self.loading_dialog = None
        self.selected_mode = ''
        self.selected_point_index = -1
        self._last_base_clamp_report = None

        self.mode_specs = [
            {'label': 'Base (YZ平面): 点を3個以上から生成', 'min_points': 3, 'requires_surface': None},
            {'label': 'Clamp (YZ平面): 点を3個以上から生成', 'min_points': 3, 'requires_surface': None},
            {'label': 'Clamp (XY平面): 点を3個以上から生成', 'min_points': 3, 'requires_surface': None},
            {'label': 'Clamp (XZ平面): 点を3個以上から生成', 'min_points': 3, 'requires_surface': None},
            {'label': 'U Surface: 点を3個以上から面を生成', 'min_points': 3, 'requires_surface': None},
            {'label': 'U Side: 点を4つ以上から生成（U Surface必要）', 'min_points': 4, 'requires_surface': 'U Surface: 点を3個以上から面を生成'},
            {'label': 'V Surface', 'min_points': 3, 'requires_surface': None},
            {'label': 'V Side', 'min_points': 4, 'requires_surface': 'V Surface'},
            {'label': 'W Surface', 'min_points': 3, 'requires_surface': None},
            {'label': 'W Side', 'min_points': 4, 'requires_surface': 'W Surface'},
            {'label': '遠位Clamp Surface', 'min_points': 3, 'requires_surface': None},
        ]

        self.mode_colors = {
            'Base (YZ平面): 点を3個以上から生成': '#f2b705',
            'Clamp (YZ平面): 点を3個以上から生成': '#00c2ff',
            'Clamp (XY平面): 点を3個以上から生成': '#3ddc97',
            'Clamp (XZ平面): 点を3個以上から生成': '#8c7bff',
            'U Surface: 点を3個以上から面を生成': '#ff4d6d',
            'U Side: 点を4つ以上から生成（U Surface必要）': '#ff9e00',
            'V Surface': '#90be6d',
            'V Side': '#43aa8b',
            'W Surface': '#577590',
            'W Side': '#4d908e',
            '遠位Clamp Surface': '#a78bfa',
        }
        self.mode_points = {spec['label']: [] for spec in self.mode_specs}

        self.label_clamp_yz = 'Clamp (YZ平面): 点を3個以上から生成'
        self.label_clamp_xy = 'Clamp (XY平面): 点を3個以上から生成'
        self.label_clamp_xz = 'Clamp (XZ平面): 点を3個以上から生成'
        self.label_clamp_lmn = 'Clamp LMN (自動生成)'
        self.clamp_lmn_color = '#ff7f50'
        self.auto_xy_plane_color = '#ff66cc'

        self.plane_modes = {
            'Base (YZ平面): 点を3個以上から生成',
            self.label_clamp_yz,
            self.label_clamp_xy,
            self.label_clamp_xz,
            'U Surface: 点を3個以上から面を生成',
            'V Surface',
            'W Surface',
            '遠位Clamp Surface',
        }
        self.side_surface_map = {
            'U Side: 点を4つ以上から生成（U Surface必要）': 'U Surface: 点を3個以上から面を生成',
            'V Side': 'V Surface',
            'W Side': 'W Surface',
        }

        self.stl_actor_name = 'stl_model'
        self.background_color = '#2a2f38'
        self.model_color = '#d9dbe0'
        self._load_visual_settings()

        root_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        title = QLabel('0点校正')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        self.load_btn = QPushButton('STLを読み込む')
        self.reload_btn = QPushButton('キャッシュから再読込')
        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.reload_btn)

        self.bg_color_btn = QPushButton('背景色を変更')
        self.model_color_btn = QPushButton('3Dモデル色を変更')
        self.bg_color_preview = QLabel('  背景色')
        self.model_color_preview = QLabel('  モデル色')
        self.bg_color_preview.setFixedHeight(22)
        self.model_color_preview.setFixedHeight(22)
        left_layout.addWidget(self.bg_color_btn)
        left_layout.addWidget(self.bg_color_preview)
        left_layout.addWidget(self.model_color_btn)
        left_layout.addWidget(self.model_color_preview)
        self._update_color_preview_labels()

        self.path_label = QLabel('未読み込み')
        self.path_label.setWordWrap(True)
        self.path_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_layout.addWidget(self.path_label)

        mode_group_box = QGroupBox('生成モード')
        mode_group_layout = QVBoxLayout(mode_group_box)
        self.mode_button_group = QButtonGroup(self)
        self.mode_radio_buttons = {}

        mode_container = QWidget()
        mode_container_layout = QVBoxLayout(mode_container)
        mode_container_layout.setContentsMargins(4, 4, 4, 4)

        for i, spec in enumerate(self.mode_specs):
            text = spec['label']
            rb = QRadioButton(text)
            if i == 0:
                rb.setChecked(True)
                self.selected_mode = text
            self.mode_button_group.addButton(rb)
            self.mode_radio_buttons[text] = rb
            mode_container_layout.addWidget(rb)

        mode_container_layout.addStretch()

        mode_scroll = QScrollArea()
        mode_scroll.setWidgetResizable(True)
        mode_scroll.setWidget(mode_container)
        mode_scroll.setMinimumHeight(180)
        mode_scroll.setMaximumHeight(260)
        mode_group_layout.addWidget(mode_scroll)
        self.mode_scroll = mode_scroll
        self._update_mode_button_highlight()

        left_layout.addWidget(mode_group_box)

        self.point_add_btn = QPushButton('点追加モード: OFF')
        self.point_add_btn.setCheckable(True)
        self.point_add_btn.setEnabled(False)
        left_layout.addWidget(self.point_add_btn)

        self.point_list = QListWidget()
        self.point_list.setMinimumHeight(120)
        self.point_list.setMaximumHeight(220)
        left_layout.addWidget(self.point_list)

        self.delete_selected_btn = QPushButton('選択点を削除')
        self.delete_selected_btn.setEnabled(False)
        self.delete_last_btn = QPushButton('最後の点を削除')
        self.delete_last_btn.setEnabled(False)
        self.clear_mode_btn = QPushButton('現在モードの点を全削除')
        self.clear_mode_btn.setEnabled(False)
        left_layout.addWidget(self.delete_selected_btn)
        left_layout.addWidget(self.delete_last_btn)
        left_layout.addWidget(self.clear_mode_btn)

        self.export_points_btn = QPushButton('点データをエクスポート')
        self.import_points_btn = QPushButton('点データをインポート')
        left_layout.addWidget(self.export_points_btn)
        left_layout.addWidget(self.import_points_btn)

        self.metrics_label = QLabel('距離計算\n- Clamp XY平面→U軸: --\n- W円面→Clamp中心面: --\n- CseXと直動X: --\n- CseZ軸と直動Z軸: --\n- CseZ軸と直動Z軸: --')
        self.metrics_label.setWordWrap(True)
        self._apply_theme_dependent_styles()
        left_layout.addWidget(self.metrics_label)

        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(340)

        right_layout = QVBoxLayout()
        if HAS_PYVISTA:
            self.plotter = QtInteractor(self)
            self._apply_plotter_background()
            self._configure_lights()
            self.plotter.add_text('STLを読み込んでください', position='upper_left', font_size=10)
            right_layout.addWidget(self.plotter.interactor)
        else:
            self.plotter = None
            fallback = QLabel('pyvista / pyvistaqt が未インストールです。')
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(fallback)

        top_layout.addWidget(left_scroll, 1)
        top_layout.addLayout(right_layout, 3)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText('読み込みログ')
        self.log_view.setMinimumHeight(160)
        self.log_view.setMaximumHeight(240)

        root_layout.addLayout(top_layout, 4)
        root_layout.addWidget(self.log_view, 1)

        self.load_btn.clicked.connect(self.on_pick_file)
        self.reload_btn.clicked.connect(self.on_reload_cached)
        self.bg_color_btn.clicked.connect(self._pick_background_color)
        self.model_color_btn.clicked.connect(self._pick_model_color)
        self.mode_button_group.buttonToggled.connect(self._on_mode_toggled)
        self.point_add_btn.toggled.connect(self._on_point_add_toggled)
        self.point_list.currentRowChanged.connect(self._on_point_list_selection_changed)
        self.delete_selected_btn.clicked.connect(self._delete_selected_point)
        self.delete_last_btn.clicked.connect(self._delete_last_point)
        self.clear_mode_btn.clicked.connect(self._clear_mode_points)
        self.export_points_btn.clicked.connect(self._on_export_points)
        self.import_points_btn.clicked.connect(self._on_import_points)

        cached = self._get_cached_stl_path()
        if cached and os.path.exists(cached):
            self._append_log(f'キャッシュ検出: {cached}')
            self._start_load(cached, show_dialog=True)
        self._refresh_point_list()
        self._log_mode_requirement()

    def _append_log(self, text: str):
        self.log_view.append(text)

    def _load_visual_settings(self):
        settings = load_settings() or {}
        tab1 = settings.get('tab1') or {}
        bg = tab1.get('background_color')
        model = tab1.get('model_color')

        if isinstance(bg, str) and QColor(bg).isValid():
            self.background_color = QColor(bg).name()
        if isinstance(model, str) and QColor(model).isValid():
            self.model_color = QColor(model).name()

    def _save_visual_settings(self):
        settings = load_settings() or {}
        tab1 = settings.setdefault('tab1', {})
        tab1['background_color'] = self.background_color
        tab1['model_color'] = self.model_color
        save_settings(settings)

    def _update_color_preview_labels(self):
        text_color = '#111111'
        if QColor(self.background_color).lightness() < 128:
            text_color = '#f7f7f7'
        self.bg_color_preview.setStyleSheet(
            f'background-color: {self.background_color}; color: {text_color}; border: 1px solid #666666; border-radius: 4px;'
        )

        text_color = '#111111'
        if QColor(self.model_color).lightness() < 128:
            text_color = '#f7f7f7'
        self.model_color_preview.setStyleSheet(
            f'background-color: {self.model_color}; color: {text_color}; border: 1px solid #666666; border-radius: 4px;'
        )

    def _background_top_color(self):
        base = QColor(self.background_color)
        if not base.isValid():
            return '#12161d'
        return base.darker(190).name()

    def _apply_plotter_background(self):
        if self.plotter is None:
            return
        self.plotter.set_background(self.background_color, top=self._background_top_color())

    def _apply_model_color(self):
        if self.plotter is None:
            return
        try:
            actor = self.plotter.renderer.actors.get(self.stl_actor_name)
        except Exception:
            actor = None
        if actor is None:
            return

        c = QColor(self.model_color)
        if not c.isValid():
            return
        actor.GetProperty().SetColor(c.redF(), c.greenF(), c.blueF())
        self.plotter.render()

    def _pick_background_color(self):
        chosen = QColorDialog.getColor(QColor(self.background_color), self, '背景色を選択')
        if not chosen.isValid():
            return
        self.background_color = chosen.name()
        self._update_color_preview_labels()
        self._apply_plotter_background()
        self._save_visual_settings()
        self._append_log(f'背景色を変更しました: {self.background_color}')

    def _pick_model_color(self):
        chosen = QColorDialog.getColor(QColor(self.model_color), self, '3Dモデル色を選択')
        if not chosen.isValid():
            return
        self.model_color = chosen.name()
        self._update_color_preview_labels()
        self._apply_model_color()
        self._save_visual_settings()
        self._append_log(f'3Dモデル色を変更しました: {self.model_color}')

    def _is_dark_mode_enabled(self) -> bool:
        settings = load_settings() or {}
        return bool(settings.get('dark', False))

    def _apply_theme_dependent_styles(self):
        is_dark = self._is_dark_mode_enabled()
        if is_dark:
            self.metrics_label.setStyleSheet('font-size: 16px; font-weight: 600; color: #f6f7fb;')
            if hasattr(self, 'mode_scroll'):
                self.mode_scroll.setStyleSheet('QScrollArea { border: 1px solid #4b5568; border-radius: 8px; background: #1a2231; }')
        else:
            self.metrics_label.setStyleSheet('font-size: 16px; font-weight: 600; color: #273043;')
            if hasattr(self, 'mode_scroll'):
                self.mode_scroll.setStyleSheet('QScrollArea { border: 1px solid #c7ced8; border-radius: 8px; background: #f7f9fc; }')

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self._apply_theme_dependent_styles()
            self._update_mode_button_highlight()

    def _on_mode_toggled(self, button, checked: bool):
        if not checked:
            return
        self.selected_mode = button.text()
        self._update_mode_button_highlight()
        self._append_log(f'モード選択: {self.selected_mode}')
        self.selected_point_index = -1
        self._refresh_point_list()
        self._update_point_buttons()
        self._update_mode_visual_focus()
        self._log_mode_requirement()
        self._save_points_cache()

    def _update_mode_button_highlight(self):
        is_dark = self._is_dark_mode_enabled()
        for mode, rb in self.mode_radio_buttons.items():
            mode_color = self.mode_colors.get(mode, '#8a8f98')
            is_selected = (mode == self.selected_mode)
            if is_selected:
                if is_dark:
                    rb.setStyleSheet(
                        f'''
                        QRadioButton {{
                            color: #f8fafc;
                            font-weight: 700;
                            border: 1px solid #5e6b80;
                            border-left: 8px solid {mode_color};
                            border-radius: 8px;
                            background-color: #1f2633;
                            padding: 8px 10px;
                        }}
                        QRadioButton::indicator {{
                            width: 18px;
                            height: 18px;
                        }}
                        QRadioButton::indicator:checked {{
                            background-color: {mode_color};
                            border: 2px solid #f8fafc;
                            border-radius: 9px;
                        }}
                        '''
                    )
                else:
                    rb.setStyleSheet(
                        f'''
                        QRadioButton {{
                            color: #1f2a37;
                            font-weight: 700;
                            border: 1px solid #b8c2cf;
                            border-left: 8px solid {mode_color};
                            border-radius: 8px;
                            background-color: #ffffff;
                            padding: 8px 10px;
                        }}
                        QRadioButton::indicator {{
                            width: 18px;
                            height: 18px;
                        }}
                        QRadioButton::indicator:checked {{
                            background-color: {mode_color};
                            border: 2px solid #ffffff;
                            border-radius: 9px;
                        }}
                        '''
                    )
            else:
                if is_dark:
                    rb.setStyleSheet(
                        '''
                        QRadioButton {
                            color: #d5d9e2;
                            font-weight: 500;
                            border: 1px solid #3b4252;
                            border-left: 8px solid transparent;
                            border-radius: 8px;
                            background-color: #171c27;
                            padding: 8px 10px;
                        }
                        QRadioButton::indicator {
                            width: 18px;
                            height: 18px;
                        }
                        QRadioButton::indicator:unchecked {
                            background-color: #171c27;
                            border: 2px solid #7a8190;
                            border-radius: 9px;
                        }
                        QRadioButton:hover {
                            border-color: #566078;
                            background-color: #1c2330;
                        }
                        '''
                    )
                else:
                    rb.setStyleSheet(
                        '''
                        QRadioButton {
                            color: #324055;
                            font-weight: 500;
                            border: 1px solid #ccd4df;
                            border-left: 8px solid transparent;
                            border-radius: 8px;
                            background-color: #f4f7fb;
                            padding: 8px 10px;
                        }
                        QRadioButton::indicator {
                            width: 18px;
                            height: 18px;
                        }
                        QRadioButton::indicator:unchecked {
                            background-color: #ffffff;
                            border: 2px solid #97a4b4;
                            border-radius: 9px;
                        }
                        QRadioButton:hover {
                            border-color: #9faec0;
                            background-color: #eaf0f7;
                        }
                        '''
                    )

    def _set_actor_emphasis(self, actor_name: str, opacity=None, line_width=None, point_size=None):
        if self.plotter is None:
            return
        try:
            actor = self.plotter.renderer.actors.get(actor_name)
        except Exception:
            actor = None
        if actor is None:
            return
        prop = actor.GetProperty()
        if opacity is not None:
            prop.SetOpacity(float(opacity))
        if line_width is not None:
            prop.SetLineWidth(float(line_width))
        if point_size is not None:
            prop.SetPointSize(float(point_size))

    def _update_mode_visual_focus(self):
        if self.plotter is None:
            return

        for spec in self.mode_specs:
            mode = spec['label']
            is_selected_mode = (mode == self.selected_mode)

            self._set_actor_emphasis(
                f'mode_points::{mode}',
                opacity=1.0 if is_selected_mode else 0.38,
                point_size=22 if is_selected_mode else 9,
            )

            if mode in self.plane_modes:
                self._set_actor_emphasis(
                    f'mode_plane::{mode}',
                    opacity=0.58 if is_selected_mode else 0.08,
                )
            elif mode in self.side_surface_map:
                self._set_actor_emphasis(
                    f'mode_side_base::{mode}',
                    opacity=0.45 if is_selected_mode else 0.07,
                )
                self._set_actor_emphasis(
                    f'mode_circle::{mode}',
                    opacity=1.0 if is_selected_mode else 0.32,
                    line_width=9 if is_selected_mode else 3,
                )
                self._set_actor_emphasis(
                    f'mode_side_axis::{mode}',
                    opacity=1.0 if is_selected_mode else 0.35,
                    line_width=6 if is_selected_mode else 2,
                )

        try:
            self.plotter.remove_actor('selected_point')
        except Exception:
            pass

        self.plotter.render()

    def _get_mode_spec(self, mode_label: str):
        for spec in self.mode_specs:
            if spec['label'] == mode_label:
                return spec
        return {'label': mode_label, 'min_points': 0, 'requires_surface': None}

    def _log_mode_requirement(self):
        spec = self._get_mode_spec(self.selected_mode)
        count = len(self.mode_points.get(self.selected_mode, []))
        deficit = max(0, int(spec.get('min_points', 0)) - count)

        if deficit > 0:
            self._append_log(f'{self.selected_mode}: あと {deficit} 点必要です。')
        else:
            self._append_log(f'{self.selected_mode}: 点数条件を満たしています。')

        required_surface = spec.get('requires_surface')
        if required_surface:
            surface_points = len(self.mode_points.get(required_surface, []))
            if surface_points < 3:
                self._append_log(f'{self.selected_mode}: 必要なSurfaceが不足しています ({required_surface}: {surface_points}/3点)')

        cyz = len(self.mode_points.get(self.label_clamp_yz, []))
        cxy = len(self.mode_points.get(self.label_clamp_xy, []))
        cxz = len(self.mode_points.get(self.label_clamp_xz, []))
        if cyz < 3 or cxy < 3 or cxz < 3:
            self._append_log(
                f'{self.label_clamp_lmn}: Clamp平面点が不足 (YZ:{cyz}/3, XY:{cxy}/3, XZ:{cxz}/3)'
            )
        else:
            self._append_log(f'{self.label_clamp_lmn}: 生成条件を満たしています。')

    def _on_point_add_toggled(self, checked: bool):
        if checked and self.current_mesh is None:
            self._append_log('先にSTLを読み込んでください。')
            self.point_add_btn.setChecked(False)
            return
        self.point_add_btn.setText('点追加モード: ON' if checked else '点追加モード: OFF')
        self._append_log('点追加モードをONにしました。左クリックで点を追加します。' if checked else '点追加モードをOFFにしました。')

    def _refresh_point_list(self):
        self.point_list.blockSignals(True)
        self.point_list.clear()
        points = self.mode_points.get(self.selected_mode, [])
        for idx, p in enumerate(points):
            self.point_list.addItem(f'{idx + 1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})')

        if 0 <= self.selected_point_index < len(points):
            self.point_list.setCurrentRow(self.selected_point_index)
        else:
            self.selected_point_index = -1
        self.point_list.blockSignals(False)

    def _on_point_list_selection_changed(self, row: int):
        self.selected_point_index = row
        self._update_point_buttons()
        self._update_selected_point_actor()

    def _update_selected_point_actor(self):
        if self.plotter is None:
            return
        try:
            self.plotter.remove_actor('selected_point')
        except Exception:
            pass

        points = self.mode_points.get(self.selected_mode, [])
        if 0 <= self.selected_point_index < len(points):
            sel = np.array([points[self.selected_point_index]], dtype=float)
            self.plotter.add_mesh(
                pv.PolyData(sel),
                name='selected_point',
                color='#ffff66',
                point_size=18,
                render_points_as_spheres=True,
                style='points',
                pickable=False,
                reset_camera=False,
                render=False,
            )
        self.plotter.render()

    def _update_point_buttons(self):
        points = self.mode_points.get(self.selected_mode, [])
        has_points = len(points) > 0
        has_selected = 0 <= self.selected_point_index < len(points)
        self.delete_selected_btn.setEnabled(has_selected)
        self.delete_last_btn.setEnabled(has_points)
        self.clear_mode_btn.setEnabled(has_points)

    def _delete_selected_point(self):
        points = self.mode_points.get(self.selected_mode, [])
        if not (0 <= self.selected_point_index < len(points)):
            return
        removed = points.pop(self.selected_point_index)
        self._append_log(f'点を削除: ({removed[0]:.2f}, {removed[1]:.2f}, {removed[2]:.2f}) [{self.selected_mode}]')
        self.selected_point_index = -1
        self._refresh_point_list()
        self._update_point_buttons()
        self._render_mode_points()
        self._log_mode_requirement()
        self._save_points_cache()

    def _delete_last_point(self):
        points = self.mode_points.get(self.selected_mode, [])
        if not points:
            return
        removed = points.pop()
        self._append_log(f'最後の点を削除: ({removed[0]:.2f}, {removed[1]:.2f}, {removed[2]:.2f}) [{self.selected_mode}]')
        self.selected_point_index = -1
        self._refresh_point_list()
        self._update_point_buttons()
        self._render_mode_points()
        self._log_mode_requirement()
        self._save_points_cache()

    def _clear_mode_points(self):
        count = len(self.mode_points.get(self.selected_mode, []))
        self.mode_points[self.selected_mode] = []
        self.selected_point_index = -1
        self._append_log(f'{self.selected_mode}: {count}点を削除しました。')
        self._refresh_point_list()
        self._update_point_buttons()
        self._render_mode_points()
        self._log_mode_requirement()
        self._save_points_cache()

    def _get_cached_stl_path(self) -> str:
        settings = load_settings() or {}
        return str((settings.get('tab1') or {}).get('stl_path') or '')

    def _save_cached_stl_path(self, path: str):
        settings = load_settings() or {}
        tab1 = settings.setdefault('tab1', {})
        tab1['stl_path'] = path
        save_settings(settings)

    def _serialize_mode_points(self):
        serialized = {}
        for mode, points in self.mode_points.items():
            serialized[mode] = [
                [float(p[0]), float(p[1]), float(p[2])] for p in points
            ]
        return serialized

    def _deserialize_mode_points(self, raw):
        parsed = {spec['label']: [] for spec in self.mode_specs}
        if not isinstance(raw, dict):
            return parsed
        for mode, points in raw.items():
            if mode not in parsed or not isinstance(points, list):
                continue
            good = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) == 3:
                    try:
                        good.append(np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float))
                    except Exception:
                        pass
            parsed[mode] = good
        return parsed

    def _save_points_cache(self):
        settings = load_settings() or {}
        tab1 = settings.setdefault('tab1', {})
        tab1['stl_path'] = self.current_path or tab1.get('stl_path', '')
        tab1['selected_mode'] = self.selected_mode
        tab1['mode_points'] = self._serialize_mode_points()
        save_settings(settings)

    def _load_points_cache_for_current_path(self):
        settings = load_settings() or {}
        tab1 = settings.get('tab1') or {}
        if tab1.get('stl_path') != self.current_path:
            return {spec['label']: [] for spec in self.mode_specs}
        return self._deserialize_mode_points(tab1.get('mode_points'))

    def _on_export_points(self):
        filepath, _ = QFileDialog.getSaveFileName(self, '点データを保存', '', 'JSON Files (*.json)')
        if not filepath:
            return
        if not filepath.endswith('.json'):
            filepath += '.json'
        if export_points_data(self.mode_points, filepath):
            self._append_log(f'点データをエクスポート: {filepath}')
        else:
            self._append_log(f'エクスポート失敗: {filepath}')

    def _on_import_points(self):
        filepath, _ = QFileDialog.getOpenFileName(self, '点データを読み込む', '', 'JSON Files (*.json)')
        if not filepath:
            return
        imported = import_points_data(filepath)
        if not imported:
            self._append_log(f'インポート失敗: {filepath}')
            return
        imported_data = self._deserialize_mode_points(imported)
        for mode, points in imported_data.items():
            if mode in self.mode_points:
                self.mode_points[mode] = points
        self.selected_point_index = -1
        self._refresh_point_list()
        self._update_point_buttons()
        self._render_mode_points()
        self._log_mode_requirement()
        self._save_points_cache()
        self._append_log(f'点データをインポート: {filepath}')

    def on_pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'STLファイルを開く', '', 'STL Files (*.stl)')
        if not path:
            return
        self._save_cached_stl_path(path)
        self._start_load(path, show_dialog=True)

    def on_reload_cached(self):
        path = self._get_cached_stl_path()
        if not path:
            self._append_log('キャッシュされたSTLパスがありません。')
            return
        if not os.path.exists(path):
            self._append_log(f'キャッシュパスが存在しません: {path}')
            return
        self._start_load(path, show_dialog=True)

    def _start_load(self, path: str, show_dialog: bool):
        self.current_path = path
        self.path_label.setText(path)
        self._append_log(f'読込要求: {path}')

        if show_dialog:
            self.loading_dialog = LoadingDialog('STL 読み込み中')
            self.loading_dialog.append_log(f'対象ファイル: {path}')
            self.loading_dialog.show()

        self.load_btn.setEnabled(False)
        self.reload_btn.setEnabled(False)

        self.thread = QThread(self)
        self.worker = STLLoadWorker(path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._on_mesh_loaded)
        self.worker.error.connect(self._on_load_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _configure_lights(self):
        if self.plotter is None:
            return
        self.plotter.remove_all_lights()

        key = pv.Light(position=(3.0, 2.0, 2.5), focal_point=(0.0, 0.0, 0.0), color='white', intensity=1.0)
        fill = pv.Light(position=(-2.0, -1.5, 1.5), focal_point=(0.0, 0.0, 0.0), color='#cfd7ff', intensity=0.45)
        rim = pv.Light(position=(-1.5, 2.5, -2.0), focal_point=(0.0, 0.0, 0.0), color='#fff1d6', intensity=0.35)

        self.plotter.add_light(key)
        self.plotter.add_light(fill)
        self.plotter.add_light(rim)

    def _is_mode_ready_for_geometry(self, mode_label: str) -> bool:
        spec = self._get_mode_spec(mode_label)
        points = self.mode_points.get(mode_label, [])
        if len(points) < int(spec.get('min_points', 0)):
            return False
        required_surface = spec.get('requires_surface')
        if required_surface and len(self.mode_points.get(required_surface, [])) < 3:
            return False
        return True

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

    def _build_side_bottom_geometry_on_surface(self, points, plane_center, u_vec, v_vec):
        arr = np.array(points, dtype=float)
        centered = arr - plane_center

        # 点を指定平面へ投影
        normal = np.cross(u_vec, v_vec)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-9:
            return None, None, None, None
        normal = normal / n_norm
        projected = arr - np.outer(np.dot(centered, normal), normal)
        p_centered = projected - plane_center

        x = np.dot(p_centered, u_vec)
        y = np.dot(p_centered, v_vec)

        # 平面上2Dで最小二乗円フィット: (x-cx)^2 + (y-cy)^2 = r^2
        if len(x) < 3:
            return None, None, None, None
        A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        b = x * x + y * y
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None, None, None, None
        cx, cy, c0 = float(sol[0]), float(sol[1]), float(sol[2])
        rad_sq = c0 + cx * cx + cy * cy
        if rad_sq <= 0.0:
            return None, None, None, None
        radius = float(np.sqrt(rad_sq))
        if radius <= 1e-6:
            return None, None, None, None

        circle_center = plane_center + cx * u_vec + cy * v_vec

        theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        circle_points = np.array([
            circle_center + radius * (np.cos(t) * u_vec + np.sin(t) * v_vec)
            for t in theta
        ])
        circle = pv.lines_from_points(circle_points, close=True)
        disk = pv.Disc(center=circle_center, inner=0.0, outer=radius, normal=normal, r_res=1, c_res=72)
        return circle, disk, circle_center, normal

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-9:
            return None
        return v / n

    def _plane_equation_from_points(self, points):
        fit = self._fit_plane_basis(points)
        if fit is None:
            return None
        center, normal, _, _ = fit
        d = -float(np.dot(normal, center))
        return normal, d

    def _infer_xyz_from_clamp_planes(self):
        yz_pts = self.mode_points.get(self.label_clamp_yz, [])
        xy_pts = self.mode_points.get(self.label_clamp_xy, [])
        xz_pts = self.mode_points.get(self.label_clamp_xz, [])
        if len(yz_pts) < 3 or len(xy_pts) < 3 or len(xz_pts) < 3:
            return None

        eq_yz = self._plane_equation_from_points(yz_pts)
        eq_xy = self._plane_equation_from_points(xy_pts)
        eq_xz = self._plane_equation_from_points(xz_pts)
        if eq_yz is None or eq_xy is None or eq_xz is None:
            return None

        n_yz, d_yz = eq_yz
        n_xy, d_xy = eq_xy
        n_xz, d_xz = eq_xz

        # L: 3平面の交点
        A = np.vstack([n_yz, n_xy, n_xz])
        b = -np.array([d_yz, d_xy, d_xz], dtype=float)
        try:
            L = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        # XはYZ平面の法線
        x_axis = self._normalize(n_yz)
        if x_axis is None:
            return None

        # X正方向: STL点が多い側
        if self.current_mesh is not None:
            pts = np.array(self.current_mesh.points, dtype=float)
            proj = (pts - L) @ x_axis
            pos_count = int(np.count_nonzero(proj >= 0.0))
            neg_count = int(np.count_nonzero(proj < 0.0))
            if pos_count < neg_count:
                x_axis = -x_axis

        # ZはXYとYZの交線方向（Y/Z入れ替え）
        z_axis = self._normalize(np.cross(n_xy, n_yz))
        if z_axis is None:
            return None
        # 方向の符号はXZ法線と整合
        if np.dot(z_axis, n_xz) < 0:
            z_axis = -z_axis

        # 現在のY/Z定義は維持しつつ、後でX符号を反転して右手系に合わせる
        y_axis = self._normalize(np.cross(x_axis, z_axis))
        if y_axis is None:
            return None

        # 要望: 現在の座標系でX軸の正負だけ入れ替える
        x_axis = -x_axis

        return {
            'L': L,
            'x': x_axis,
            'y': y_axis,
            'z': z_axis,
            'n_yz': n_yz,
            'd_yz': d_yz,
        }

    def _check_base_clamp_yz_consistency(self):
        base_mode = 'Base (YZ平面): 点を3個以上から生成'
        base_pts = self.mode_points.get(base_mode, [])
        clamp_pts = self.mode_points.get(self.label_clamp_yz, [])
        if len(base_pts) < 3 or len(clamp_pts) < 3:
            self._last_base_clamp_report = None
            return

        base_eq = self._plane_equation_from_points(base_pts)
        clamp_eq = self._plane_equation_from_points(clamp_pts)
        if base_eq is None or clamp_eq is None:
            return

        n_base, d_base = base_eq
        n_clamp, d_clamp = clamp_eq
        cosang = float(np.clip(abs(np.dot(n_base, n_clamp)), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cosang)))
        # 要件: 一致判定は角度のみ
        if angle_deg > 5.0:
            report = f'mismatch:{angle_deg:.2f}'
            if report != self._last_base_clamp_report:
                self._append_log(
                    f'注意: Base YZ と Clamp YZ が一致していない可能性 (角度差={angle_deg:.2f}deg)'
                )
                self._last_base_clamp_report = report
        else:
            report = 'ok'
            if report != self._last_base_clamp_report:
                self._append_log('Base YZ と Clamp YZ はほぼ一致しています。')
                self._last_base_clamp_report = report

    def _build_clamp_center_plane(self, center_point):
        # まず推定座標系の +Z を法線として使い、XY平面と平行を保証する
        axes = self._infer_xyz_from_clamp_planes()
        if axes is not None:
            plane_normal = axes['z']
        else:
            # フォールバック: Clamp XY の法線
            xy_pts = self.mode_points.get(self.label_clamp_xy, [])
            if len(xy_pts) < 3:
                return None
            eq_xy = self._plane_equation_from_points(xy_pts)
            if eq_xy is None:
                return None
            plane_normal, _d_xy = eq_xy

        # STLを覆うサイズ
        plane_size = 400.0
        if self.current_mesh is not None:
            b = self.current_mesh.bounds
            span = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]], dtype=float)
            max_span = float(np.max(span))
            plane_size = max(200.0, max_span * 1.6)

        return pv.Plane(
            center=np.array(center_point, dtype=float),
            direction=plane_normal,
            i_size=plane_size,
            j_size=plane_size,
            i_resolution=1,
            j_resolution=1,
        )

    def _build_clamp_lmn_geometry(self):
        yz_pts = self.mode_points.get(self.label_clamp_yz, [])
        xy_pts = self.mode_points.get(self.label_clamp_xy, [])
        xz_pts = self.mode_points.get(self.label_clamp_xz, [])
        if len(yz_pts) < 3 or len(xy_pts) < 3 or len(xz_pts) < 3:
            return None

        eq_yz = self._plane_equation_from_points(yz_pts)
        eq_xy = self._plane_equation_from_points(xy_pts)
        eq_xz = self._plane_equation_from_points(xz_pts)
        if eq_yz is None or eq_xy is None or eq_xz is None:
            return None

        n_yz, d_yz = eq_yz
        n_xy, d_xy = eq_xy
        n_xz, d_xz = eq_xz

        A = np.vstack([n_yz, n_xy, n_xz])
        b = -np.array([d_yz, d_xy, d_xz], dtype=float)
        try:
            L = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        # 指定定義:
        # LM: Clamp XY と Clamp XZ の交線方向（200mm）
        # LN: Clamp XY と Clamp YZ の交線方向（234mm）
        dir_lm = self._normalize(np.cross(n_xy, n_xz))
        dir_ln = self._normalize(np.cross(n_xy, n_yz))
        if dir_lm is None or dir_ln is None:
            return None

        # 方向を推定XYZに整合
        axes = self._infer_xyz_from_clamp_planes()
        if axes is not None:
            # 要件: LM は -X 側、LN は -Z 側へ伸ばす
            if np.dot(dir_lm, axes['x']) > 0:
                dir_lm = -dir_lm
            if np.dot(dir_ln, axes['z']) < 0:
                dir_ln = -dir_ln

        M = L + 200.0 * dir_lm
        N = L + 234.0 * dir_ln

        tri_points = np.array([L, M, N], dtype=float)
        tri = pv.PolyData(tri_points)
        tri.faces = np.array([3, 0, 1, 2], dtype=np.int64)

        line_lm = pv.lines_from_points(np.array([L, M], dtype=float), close=False)
        line_ln = pv.lines_from_points(np.array([L, N], dtype=float), close=False)
        return tri, line_lm, line_ln, L, M, N

    @staticmethod
    def _line_to_plane_distance(line_point, line_dir, plane_point, plane_normal):
        ld = np.array(line_dir, dtype=float)
        lp = np.array(line_point, dtype=float)
        pp = np.array(plane_point, dtype=float)
        pn = np.array(plane_normal, dtype=float)
        ld_n = np.linalg.norm(ld)
        pn_n = np.linalg.norm(pn)
        if ld_n < 1e-9 or pn_n < 1e-9:
            return None
        ld = ld / ld_n
        pn = pn / pn_n
        denom = abs(float(np.dot(ld, pn)))
        if denom > 1e-8:
            return 0.0
        return float(abs(np.dot(lp - pp, pn)))

    @staticmethod
    def _plane_to_plane_distance(plane1_point, plane1_normal, plane2_point, plane2_normal):
        p1 = np.array(plane1_point, dtype=float)
        n1 = np.array(plane1_normal, dtype=float)
        p2 = np.array(plane2_point, dtype=float)
        n2 = np.array(plane2_normal, dtype=float)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-9 or n2_norm < 1e-9:
            return None
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # 非平行なら2平面は交わるため最短距離は0
        if np.linalg.norm(np.cross(n1, n2)) > 1e-8:
            return 0.0
        return float(abs(np.dot(p1 - p2, n1)))

    @staticmethod
    def _point_to_plane_distance(point, plane_point, plane_normal):
        p = np.array(point, dtype=float)
        pp = np.array(plane_point, dtype=float)
        pn = np.array(plane_normal, dtype=float)
        pn_n = np.linalg.norm(pn)
        if pn_n < 1e-9:
            return None
        pn = pn / pn_n
        return float(abs(np.dot(p - pp, pn)))

    @staticmethod
    def _project_point_to_plane(point, plane_point, plane_normal):
        p = np.array(point, dtype=float)
        pp = np.array(plane_point, dtype=float)
        pn = np.array(plane_normal, dtype=float)
        pn_n = np.linalg.norm(pn)
        if pn_n < 1e-9:
            return None
        pn = pn / pn_n
        signed = float(np.dot(p - pp, pn))
        return p - signed * pn

    @staticmethod
    def _line_to_plane_measure_points(line_point, line_dir, plane_point, plane_normal):
        lp = np.array(line_point, dtype=float)
        ld = np.array(line_dir, dtype=float)
        pp = np.array(plane_point, dtype=float)
        pn = np.array(plane_normal, dtype=float)

        ld_n = np.linalg.norm(ld)
        pn_n = np.linalg.norm(pn)
        if ld_n < 1e-9 or pn_n < 1e-9:
            return None, None
        ld = ld / ld_n
        pn = pn / pn_n

        den = float(np.dot(ld, pn))
        if abs(den) > 1e-8:
            t = float(np.dot(pp - lp, pn) / den)
            p = lp + t * ld
            return p, p

        q = lp - float(np.dot(lp - pp, pn)) * pn
        return lp, q

    @staticmethod
    def _point_in_triangle_3d(point, a, b, c, tri_normal):
        p = np.array(point, dtype=float)
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)
        n = np.array(tri_normal, dtype=float)

        if np.dot(np.cross(b - a, p - a), n) < -1e-8:
            return False
        if np.dot(np.cross(c - b, p - b), n) < -1e-8:
            return False
        if np.dot(np.cross(a - c, p - c), n) < -1e-8:
            return False
        return True

    @staticmethod
    def _closest_points_line_segment(line_point, line_dir, seg_a, seg_b):
        p0 = np.array(line_point, dtype=float)
        u = np.array(line_dir, dtype=float)
        a = np.array(seg_a, dtype=float)
        b = np.array(seg_b, dtype=float)

        u_norm = np.linalg.norm(u)
        if u_norm < 1e-9:
            return None, None, None
        u = u / u_norm

        v = b - a
        c = float(np.dot(v, v))
        if c < 1e-9:
            s = float(np.dot(a - p0, u))
            p_line = p0 + s * u
            p_seg = a
            return p_line, p_seg, float(np.linalg.norm(p_line - p_seg))

        w0 = p0 - a
        b_uv = float(np.dot(u, v))
        d = float(np.dot(u, w0))
        e = float(np.dot(v, w0))
        denom = c - b_uv * b_uv

        if abs(denom) < 1e-9:
            t = float(np.clip(e / c, 0.0, 1.0))
        else:
            t = float(np.clip((b_uv * d - e) / denom, 0.0, 1.0))

        s = b_uv * t - d
        p_line = p0 + s * u
        p_seg = a + t * v
        return p_line, p_seg, float(np.linalg.norm(p_line - p_seg))

    def _line_to_triangle_distance(self, line_point, line_dir, tri_a, tri_b, tri_c):
        lp = np.array(line_point, dtype=float)
        ld = np.array(line_dir, dtype=float)
        a = np.array(tri_a, dtype=float)
        b = np.array(tri_b, dtype=float)
        c = np.array(tri_c, dtype=float)

        ld_n = np.linalg.norm(ld)
        if ld_n < 1e-9:
            return None, None, None
        ld = ld / ld_n

        tri_n = self._normalize(np.cross(b - a, c - a))
        if tri_n is None:
            return None, None, None

        den = float(np.dot(ld, tri_n))
        if abs(den) > 1e-8:
            t = float(np.dot(a - lp, tri_n) / den)
            hit = lp + t * ld
            if self._point_in_triangle_3d(hit, a, b, c, tri_n):
                return 0.0, hit, hit

        best_d = None
        best_line = None
        best_tri = None
        for s0, s1 in ((a, b), (b, c), (c, a)):
            p_line, p_seg, dist = self._closest_points_line_segment(lp, ld, s0, s1)
            if dist is None:
                continue
            if best_d is None or dist < best_d:
                best_d = dist
                best_line = p_line
                best_tri = p_seg

        return best_d, best_line, best_tri

    def _render_mode_points(self):
        if self.plotter is None:
            return

        self._check_base_clamp_yz_consistency()

        # 点再描画でカメラがリセットされないように現在位置を保持
        camera_position = self.plotter.camera_position

        diag = 400.0
        if self.current_mesh is not None:
            b = self.current_mesh.bounds
            diag = float(np.linalg.norm(np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]], dtype=float)))
        xyz_axis_len = float(np.clip(diag * 0.12, 50.0, 250.0))
        side_axis_len = float(np.clip(diag * 0.35, 120.0, 500.0))

        # 既存ポイントアクタを消して再描画
        for spec in self.mode_specs:
            mode = spec['label']
            for actor_name in (
                f"mode_points::{mode}",
                f"mode_plane::{mode}",
                f"mode_circle::{mode}",
                f"mode_edge1::{mode}",
                f"mode_edge2::{mode}",
                f"mode_side_base::{mode}",
                f"mode_side_axis::{mode}",
            ):
                try:
                    self.plotter.remove_actor(actor_name)
                except Exception:
                    pass

        for auto_name in (
            'mode_plane::auto_lmn',
            'mode_plane::clamp_center',
            'mode_edge1::auto_lmn',
            'mode_edge2::auto_lmn',
            'mode_points::auto_lmn',
            'mode_labels::auto_lmn',
            'mode_axis::x',
            'mode_axis::y',
            'mode_axis::z',
            'mode_labels::xyz',
            'mode_labels::distances',
            'mode_measure::u_clamp_xy',
            'mode_measure::w_clamp_center',
            'mode_measure_points::u_clamp_xy',
            'mode_measure_points::w_clamp_center',
            'selected_point',
        ):
            try:
                self.plotter.remove_actor(auto_name)
            except Exception:
                pass

        side_axis_info = {}
        side_circle_plane_info = {}

        for spec in self.mode_specs:
            mode = spec['label']
            points = self.mode_points.get(mode, [])
            if not points:
                continue

            is_selected_mode = (mode == self.selected_mode)
            arr = np.array(points, dtype=float)
            self.plotter.add_mesh(
                pv.PolyData(arr),
                name=f"mode_points::{mode}",
                color=self.mode_colors.get(mode, '#ffffff'),
                point_size=16 if is_selected_mode else 12,
                render_points_as_spheres=True,
                style='points',
                pickable=False,
                reset_camera=False,
                render=False,
            )

            if not self._is_mode_ready_for_geometry(mode):
                continue

            if mode in self.plane_modes:
                plane = self._build_plane_from_points(points)
                if plane is not None:
                    self.plotter.add_mesh(
                        plane,
                        name=f"mode_plane::{mode}",
                        color=self.mode_colors.get(mode, '#ffffff'),
                        opacity=0.45 if is_selected_mode else 0.20,
                        pickable=False,
                        reset_camera=False,
                        render=False,
                        show_edges=False,
                        smooth_shading=True,
                    )
            elif mode in self.side_surface_map:
                surf_mode = self.side_surface_map[mode]
                surf_fit = self._fit_plane_basis(self.mode_points.get(surf_mode, []))
                if surf_fit is None:
                    continue

                s_center, _s_normal, s_u, s_v = surf_fit
                circle, disk, c_center, c_normal = self._build_side_bottom_geometry_on_surface(points, s_center, s_u, s_v)

                if disk is not None:
                    self.plotter.add_mesh(
                        disk,
                        name=f"mode_side_base::{mode}",
                        color=self.mode_colors.get(mode, '#ffffff'),
                        opacity=0.35 if is_selected_mode else 0.18,
                        pickable=False,
                        reset_camera=False,
                        render=False,
                        smooth_shading=True,
                    )
                if circle is not None:
                    self.plotter.add_mesh(
                        circle,
                        name=f"mode_circle::{mode}",
                        color=self.mode_colors.get(mode, '#ffffff'),
                        line_width=7 if is_selected_mode else 4,
                        pickable=False,
                        reset_camera=False,
                        render=False,
                    )

                if c_center is not None and c_normal is not None:
                    side_circle_plane_info[mode] = {'point': np.array(c_center, dtype=float), 'normal': np.array(c_normal, dtype=float)}
                    # 法線の向きを点群の重心方向に合わせる
                    arr_pts = np.array(points, dtype=float)
                    points_centroid = np.mean(arr_pts, axis=0)
                    normal_vec = self._normalize(np.array(c_normal, dtype=float))
                    centroid_dir = self._normalize(points_centroid - np.array(c_center, dtype=float))
                    if normal_vec is not None and centroid_dir is not None:
                        if np.dot(normal_vec, centroid_dir) < 0:
                            normal_vec = -normal_vec
                    axis_dir = self._normalize(-normal_vec) if normal_vec is not None else None
                    if axis_dir is not None:
                        p0 = np.array(c_center, dtype=float)
                        p1 = p0 + side_axis_len * axis_dir
                        side_axis_info[mode] = {'point': p0, 'dir': axis_dir}
                        self.plotter.add_mesh(
                            pv.lines_from_points(np.array([p0, p1], dtype=float), close=False),
                            name=f"mode_side_axis::{mode}",
                            color=self.mode_colors.get(mode, '#ffffff'),
                            line_width=4,
                            pickable=False,
                            reset_camera=False,
                            render=False,
                        )

        geo = self._build_clamp_lmn_geometry()
        clamp_center_plane_point = None
        clamp_center_plane_normal = None
        lmn_plane_point = None
        lmn_plane_normal = None

        if geo is not None:
            tri, line_lm, line_ln, l_pt, m_pt, n_pt = geo
            lmn_center = (np.array(l_pt) + np.array(n_pt)) / 2.0  # Clamp Center は LN 中点

            lmn_plane_point = np.array(l_pt, dtype=float)
            lmn_plane_normal = self._normalize(
                np.cross(np.array(m_pt, dtype=float) - np.array(l_pt, dtype=float), np.array(n_pt, dtype=float) - np.array(l_pt, dtype=float))
            )

            clamp_center_plane = self._build_clamp_center_plane(lmn_center)
            if clamp_center_plane is not None:
                clamp_center_plane_point = np.array(lmn_center, dtype=float)
                axes_for_plane = self._infer_xyz_from_clamp_planes()
                if axes_for_plane is not None:
                    clamp_center_plane_normal = axes_for_plane['z']
                else:
                    eq_xy = self._plane_equation_from_points(self.mode_points.get(self.label_clamp_xy, []))
                    if eq_xy is not None:
                        clamp_center_plane_normal = eq_xy[0]

                self.plotter.add_mesh(
                    clamp_center_plane,
                    name='mode_plane::clamp_center',
                    color=self.auto_xy_plane_color,
                    opacity=0.16,
                    pickable=False,
                    reset_camera=False,
                    render=False,
                    show_edges=False,
                    smooth_shading=True,
                )

            self.plotter.add_mesh(
                tri,
                name='mode_plane::auto_lmn',
                color=self.clamp_lmn_color,
                opacity=0.22,
                pickable=False,
                reset_camera=False,
                render=False,
                show_edges=False,
                smooth_shading=True,
            )
            self.plotter.add_mesh(
                line_lm,
                name='mode_edge1::auto_lmn',
                color=self.clamp_lmn_color,
                line_width=4,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            self.plotter.add_mesh(
                line_ln,
                name='mode_edge2::auto_lmn',
                color=self.clamp_lmn_color,
                line_width=4,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            self.plotter.add_mesh(
                pv.PolyData(np.array([l_pt, m_pt, n_pt], dtype=float)),
                name='mode_points::auto_lmn',
                color=self.clamp_lmn_color,
                point_size=14,
                render_points_as_spheres=True,
                style='points',
                pickable=False,
                reset_camera=False,
                render=False,
            )
            self.plotter.add_point_labels(
                np.array([l_pt, m_pt, n_pt], dtype=float),
                ['L', 'M', 'N'],
                name='mode_labels::auto_lmn',
                text_color=self.clamp_lmn_color,
                font_size=14,
                always_visible=True,
                fill_shape=True,
                shape_opacity=0.18,
                margin=4,
                reset_camera=False,
                render=False,
            )

        # 距離計算
        u_axis = side_axis_info.get('U Side: 点を4つ以上から生成（U Surface必要）')
        w_circle_plane = side_circle_plane_info.get('W Side')

        clamp_xy_fit = self._fit_plane_basis(self.mode_points.get(self.label_clamp_xy, []))
        clamp_xy_plane_point = clamp_xy_fit[0] if clamp_xy_fit is not None else None
        clamp_xy_plane_normal = clamp_xy_fit[1] if clamp_xy_fit is not None else None

        d_u_clamp_xy = None
        u_clamp_xy_p0 = None
        u_clamp_xy_p1 = None
        d_w_circle_clamp_center = None
        angle_v_y = None
        angle_w_z = None
        angle_cse_x = None

        if u_axis is not None and clamp_xy_plane_point is not None and clamp_xy_plane_normal is not None:
            d_u_clamp_xy = self._point_to_plane_distance(
                u_axis['point'],
                clamp_xy_plane_point,
                clamp_xy_plane_normal,
            )
            u_clamp_xy_p0 = np.array(u_axis['point'], dtype=float)
            u_clamp_xy_p1 = self._project_point_to_plane(
                u_clamp_xy_p0,
                clamp_xy_plane_point,
                clamp_xy_plane_normal,
            )
        if w_circle_plane is not None and clamp_center_plane_point is not None and clamp_center_plane_normal is not None:
            d_w_circle_clamp_center = self._point_to_plane_distance(
                w_circle_plane['point'],
                clamp_center_plane_point,
                clamp_center_plane_normal,
            )

        # V ベクトルとクランプ座標系の Y 軸のなす角
        v_axis = side_axis_info.get('V Side')
        if v_axis is not None:
            v_vec = self._normalize(np.array(v_axis['dir'], dtype=float))
            axes = self._infer_xyz_from_clamp_planes()
            if v_vec is not None and axes is not None:
                y_axis = self._normalize(axes['y'])
                if y_axis is not None:
                    cos_angle = np.dot(v_vec, y_axis)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_v_y = float(np.degrees(np.arccos(np.abs(cos_angle))))

        # W ベクトルとクランプ座標系の Z 軸のなす角
        w_axis = side_axis_info.get('W Side')
        if w_axis is not None:
            w_vec = self._normalize(np.array(w_axis['dir'], dtype=float))
            axes = self._infer_xyz_from_clamp_planes()
            if w_vec is not None and axes is not None:
                z_axis = self._normalize(axes['z'])
                if z_axis is not None:
                    cos_angle = np.dot(w_vec, z_axis)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_w_z = float(np.degrees(np.arccos(np.abs(cos_angle))))

        # 遠位Clamp Surface の平面とクランプ座標系の X 軸とのなす角（CseX）
        cse_points = self.mode_points.get('遠位Clamp Surface', [])
        if cse_points:
            fit = self._fit_plane_basis(cse_points)
            axes = self._infer_xyz_from_clamp_planes()
            if fit is not None and axes is not None:
                cse_normal = self._normalize(fit[1])
                x_axis = self._normalize(axes['x'])
                if cse_normal is not None and x_axis is not None:
                    cos_angle = np.dot(cse_normal, x_axis)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_cse_x = float(np.degrees(np.arccos(np.abs(cos_angle))))

        def _fmt(v):
            return '--' if v is None else f'{v:.3f} mm'

        def _fmt_angle(v):
            return '--' if v is None else f'{v:.1f}°'

        self.metrics_label.setText(
            '距離計算\n'
            f'- Clamp XY平面→U軸: {_fmt(d_u_clamp_xy)}\n'
            f'- W円面→Clamp中心面: {_fmt(d_w_circle_clamp_center)}\n'
            f'- CseXと直動X: {_fmt_angle(angle_cse_x)}\n'
            f'- CseZ軸と直動Z軸: {_fmt_angle(angle_v_y)}\n'
            f'- CseZ軸と直動Z軸: {_fmt_angle(angle_w_z)}'
        )

        # 右画面にも距離テキストを表示
        dist_points = []
        dist_labels = []
        if u_axis is not None and d_u_clamp_xy is not None:
            dist_points.append(u_axis['point'] + 0.52 * side_axis_len * u_axis['dir'])
            dist_labels.append(f'ClampXY→U: {d_u_clamp_xy:.2f}mm')
        if w_circle_plane is not None and d_w_circle_clamp_center is not None:
            dist_points.append(w_circle_plane['point'])
            dist_labels.append(f'W円面→ClampC: {d_w_circle_clamp_center:.2f}mm')

        # 距離の測定元/測定先を明示する線分を描画
        if u_clamp_xy_p0 is not None and u_clamp_xy_p1 is not None:
            p0, p1 = u_clamp_xy_p0, u_clamp_xy_p1
            if p0 is not None and p1 is not None:
                self.plotter.add_mesh(
                    pv.lines_from_points(np.array([p0, p1], dtype=float), close=False),
                    name='mode_measure::u_clamp_xy',
                    color='#8ce99a',
                    line_width=5,
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                self.plotter.add_mesh(
                    pv.PolyData(np.array([p0, p1], dtype=float)),
                    name='mode_measure_points::u_clamp_xy',
                    color='#8ce99a',
                    point_size=11,
                    render_points_as_spheres=True,
                    style='points',
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )

        if w_circle_plane is not None and clamp_center_plane_point is not None and clamp_center_plane_normal is not None:
            w_center = np.array(w_circle_plane['point'], dtype=float)
            w_proj = self._project_point_to_plane(w_center, clamp_center_plane_point, clamp_center_plane_normal)
            if w_proj is not None:
                self.plotter.add_mesh(
                    pv.lines_from_points(np.array([w_center, w_proj], dtype=float), close=False),
                    name='mode_measure::w_clamp_center',
                    color='#ff922b',
                    line_width=6,
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                self.plotter.add_mesh(
                    pv.PolyData(np.array([w_center, w_proj], dtype=float)),
                    name='mode_measure_points::w_clamp_center',
                    color='#ff922b',
                    point_size=12,
                    render_points_as_spheres=True,
                    style='points',
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )

        if dist_points:
            self.plotter.add_point_labels(
                np.array(dist_points, dtype=float),
                dist_labels,
                name='mode_labels::distances',
                text_color='#ffe58f',
                font_size=12,
                always_visible=True,
                fill_shape=True,
                shape_opacity=0.22,
                margin=4,
                reset_camera=False,
                render=False,
            )

        # ワールド座標軸の代わりに、Clamp平面群から推定したXYZを表示（細線）
        axes = self._infer_xyz_from_clamp_planes()
        if axes is not None:
            l_pt = axes['L']
            x_end = l_pt + xyz_axis_len * axes['x']
            y_end = l_pt + xyz_axis_len * axes['y']
            z_end = l_pt + xyz_axis_len * axes['z']

            self.plotter.add_mesh(
                pv.lines_from_points(np.array([l_pt, x_end], dtype=float), close=False),
                name='mode_axis::x',
                color='#ff4d4f',
                line_width=4,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            self.plotter.add_mesh(
                pv.lines_from_points(np.array([l_pt, y_end], dtype=float), close=False),
                name='mode_axis::y',
                color='#52c41a',
                line_width=4,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            self.plotter.add_mesh(
                pv.lines_from_points(np.array([l_pt, z_end], dtype=float), close=False),
                name='mode_axis::z',
                color='#1677ff',
                line_width=4,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            self.plotter.add_point_labels(
                np.array([x_end, y_end, z_end], dtype=float),
                ['+X', '+Y', '+Z'],
                name='mode_labels::xyz',
                text_color='white',
                font_size=12,
                always_visible=True,
                fill_shape=True,
                shape_opacity=0.15,
                margin=4,
                reset_camera=False,
                render=False,
            )

        points = self.mode_points.get(self.selected_mode, [])
        if 0 <= self.selected_point_index < len(points):
            sel = np.array([points[self.selected_point_index]], dtype=float)
            self.plotter.add_mesh(
                pv.PolyData(sel),
                name='selected_point',
                color='#ffff66',
                point_size=18,
                render_points_as_spheres=True,
                style='points',
                pickable=False,
                reset_camera=False,
                render=False,
            )

        self.plotter.camera_position = camera_position
        self.plotter.render()

    def _on_surface_point_picked(self, point, *_args):
        if not self.point_add_btn.isChecked():
            return
        if point is None:
            return

        spec = self._get_mode_spec(self.selected_mode)
        required_surface = spec.get('requires_surface')
        if required_surface and len(self.mode_points.get(required_surface, [])) < 3:
            self._append_log(f'{self.selected_mode}: 先に {required_surface} を3点以上作成してください。')
            return

        p = np.array(point, dtype=float)
        self.mode_points[self.selected_mode].append(p)
        self.selected_point_index = len(self.mode_points[self.selected_mode]) - 1
        self._append_log(
            f'点追加 [{self.selected_mode}] #{self.selected_point_index + 1}: '
            f'({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})'
        )
        self._refresh_point_list()
        self._update_point_buttons()
        self._render_mode_points()
        self._log_mode_requirement()
        self._save_points_cache()

    def _on_mesh_loaded(self, mesh):
        self.current_mesh = mesh
        self._append_log('表示を更新します...')

        # キャッシュが同一STL向けにあれば復元
        self.mode_points = self._load_points_cache_for_current_path()
        settings = load_settings() or {}
        cached_selected = (settings.get('tab1') or {}).get('selected_mode')
        if isinstance(cached_selected, str) and cached_selected in self.mode_points:
            self.selected_mode = cached_selected
            for rb in self.mode_button_group.buttons():
                if rb.text() == cached_selected:
                    rb.setChecked(True)
                    break
        self.selected_point_index = -1
        self._refresh_point_list()
        self._update_point_buttons()

        if self.plotter is not None:
            # pyvista の picking は clear() では解除されないため、再設定前に明示解除する
            try:
                self.plotter.disable_picking()
            except Exception:
                pass
            self.plotter.clear()
            self._apply_plotter_background()
            self._configure_lights()
            self.plotter.add_mesh(
                mesh,
                name=self.stl_actor_name,
                color=self.model_color,
                show_edges=False,
                smooth_shading=True,
                ambient=0.15,
                diffuse=0.75,
                specular=0.35,
                specular_power=25.0,
                pickable=True,
            )
            self.plotter.hide_axes()
            self.plotter.enable_surface_point_picking(
                callback=self._on_surface_point_picked,
                left_clicking=True,
                show_point=False,
                pickable_window=False,
            )
            try:
                self._render_mode_points()
            except Exception as e:
                self._append_log(f'補助描画でエラー: {e}')

            # 補助点群が離れた位置にあっても、初期視点は必ずSTL本体に合わせる
            try:
                self.plotter.reset_camera(bounds=mesh.bounds)
            except TypeError:
                self.plotter.reset_camera()
            self.plotter.render()

        self.load_btn.setEnabled(True)
        self.reload_btn.setEnabled(True)
        self.point_add_btn.setEnabled(True)
        self._append_log('点追加: 「点追加モード」をONにして3D上を左クリックしてください。')
        self._log_mode_requirement()
        self._save_points_cache()
        if self.loading_dialog is not None:
            self.loading_dialog.append_log('完了')
            self.loading_dialog.close()
            self.loading_dialog = None

    def _on_load_error(self, msg: str):
        self._append_log(msg)
        self.load_btn.setEnabled(True)
        self.reload_btn.setEnabled(True)
        self.point_add_btn.setEnabled(self.current_mesh is not None)
        if self.loading_dialog is not None:
            self.loading_dialog.append_log(msg)
            self.loading_dialog.close()
            self.loading_dialog = None
