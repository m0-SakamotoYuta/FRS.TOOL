from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QGroupBox, QRadioButton,
    QButtonGroup, QScrollArea, QPushButton, QFileDialog, QTextEdit, QListWidget,
    QSlider, QSpinBox, QCheckBox, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QEvent
from PyQt6.QtGui import QColor
import os
import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False

from tabs.settings import (
    load_settings,
    save_settings,
    get_lighting_enabled,
    set_lighting_enabled,
    register_lighting_listener,
)


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
            mesh = mesh.extract_surface(algorithm='dataset_surface').triangulate()
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


# ===== RANSAC → ICP フィッティング（open3d）=====

def _mesh_to_o3d_pcd(mesh, voxel_size):
    """pyvista mesh → ダウンサンプリング済み open3d PointCloud（法線付き）と FPFH 特徴量。"""
    pts = np.asarray(mesh.points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd_down = pcd.voxel_down_sample(voxel_size) if voxel_size > 0 else pcd
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd_down, fpfh


def _default_voxel_size(mesh):
    """メッシュの対角長から妥当な voxel サイズ（mm）を推定する。"""
    try:
        b = mesh.bounds
        diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
    except Exception:
        diag = 100.0
    return max(diag * 0.02, 1e-3)


def ransac_icp_fit(source_mesh, target_mesh, params, log=None):
    """source（姿勢の任意領域）を target（base の任意領域）へ剛体フィッティングする。

    返り値 dict:
        transform: 4x4 (source 座標 → target 座標)
        ransac_fitness, ransac_rmse, icp_fitness, icp_rmse, voxel_size
    """
    def _log(msg):
        if callable(log):
            log(msg)

    voxel_size = float(params.get('voxel_size', 0.0) or 0.0)
    if voxel_size <= 0:
        voxel_size = _default_voxel_size(target_mesh)
        _log(f'voxel サイズを自動推定: {voxel_size:.4f} mm')

    ransac_iter = int(params.get('ransac_iter', 100000))
    icp_iter = int(params.get('icp_iter', 50))
    dist_factor = float(params.get('dist_factor', 1.5))

    _log('点群を準備中（ダウンサンプリング + FPFH 特徴量）...')
    src_down, src_fpfh = _mesh_to_o3d_pcd(source_mesh, voxel_size)
    tgt_down, tgt_fpfh = _mesh_to_o3d_pcd(target_mesh, voxel_size)

    distance_threshold = voxel_size * dist_factor
    _log(f'RANSAC グローバル位置合わせ中（反復上限 {ransac_iter}, 閾値 {distance_threshold:.4f} mm）...')
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iter, 0.999),
    )
    _log(f'  RANSAC: fitness={ransac.fitness:.4f}, inlier_rmse={ransac.inlier_rmse:.4f} mm')

    icp_threshold = voxel_size * 0.8
    _log(f'ICP 微調整中（反復上限 {icp_iter}, 閾値 {icp_threshold:.4f} mm）...')
    icp = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down, icp_threshold, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iter),
    )
    _log(f'  ICP: fitness={icp.fitness:.4f}, inlier_rmse={icp.inlier_rmse:.4f} mm')

    return {
        'transform': np.asarray(icp.transformation, dtype=float).copy(),
        'ransac_fitness': float(ransac.fitness),
        'ransac_rmse': float(ransac.inlier_rmse),
        'icp_fitness': float(icp.fitness),
        'icp_rmse': float(icp.inlier_rmse),
        'voxel_size': voxel_size,
    }


class FitWorker(QObject):
    """RANSAC→ICP フィッティング用ワーカースレッド"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, source_mesh, target_mesh, params):
        super().__init__()
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.params = params

    def run(self):
        if not HAS_OPEN3D:
            self.error.emit('open3d が見つかりません。`pip install open3d` を実行してください。')
            return
        try:
            result = ransac_icp_fit(
                self.source_mesh, self.target_mesh, self.params, log=self.log.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f'フィッティング失敗: {e}')


class Tab2Widget(QWidget):
    # サブクラス（FEAxisWidget 等）で別の設定キーへ切り替えるためのデフォルト
    SETTINGS_TOP_KEY = 'tab2'

    # 姿勢数とラベル定義（U/V/W/X/Y/Z 各軸で共通）。サブクラスで上書き可。
    POSTURE_SPECS = [
        ('姿勢1', '', 'posture1'),
        ('姿勢2', '', 'posture2'),
        ('姿勢3', '', 'posture3'),
        ('姿勢4', '', 'posture4'),
        ('姿勢5', '', 'posture5'),
    ]
    POSTURE_LABELS = ['姿勢1', '姿勢2', '姿勢3', '姿勢4', '姿勢5']

    def __init__(self, parent=None):
        super().__init__(parent)
        # 設定ファイル中のトップキー（サブクラスでオーバーライド可）
        self.settings_top_key = type(self).SETTINGS_TOP_KEY
        # FE 専用 UI（座標系名の自由入力 / 軸の太さ・長さスライダー / ボタンラベル差し替え）
        self.fe_mode = False
        # 3D テキスト用の日本語フォント（メイリオ優先）
        self._jp_font_path = self._detect_jp_font()
        # 軸ごとに独立した posture_widgets を保持: {'u': {...}, 'v': {...}, 'w': {...}}
        self.axis_data = {}
        self.visual_widgets = []
        self.lighting_checkboxes = []
        # C_world 座標系での共有カメラ姿勢（タブ間で引き継ぐ）
        self.shared_camera_world = None
        # 手動で「記録」した視点（C_world 座標系）
        self.recorded_view_world = None
        self.all_view_widget = None
        self._load_shared_camera_cache()
        self._load_recorded_view_cache()
        register_lighting_listener(self._on_global_lighting_changed)

        layout = QVBoxLayout(self)

        self.top_subtabs = QTabWidget()
        # ALL VIEW 本体は単一インスタンス。トップタブと各軸タブの先頭サブタブの
        # ホストへ、タブ切替時に付け替える（reparent）ことで完全同期させる。
        self.all_view_widget = self._create_all_view_tab()
        self.all_view_host_main = QWidget()
        _host_layout = QVBoxLayout(self.all_view_host_main)
        _host_layout.setContentsMargins(0, 0, 0, 0)

        axis_names = ['ALL VIEW', 'U axis', 'V axis', 'W axis', 'X axis', 'Y axis', 'Z axis']
        for axis_name in axis_names:
            if axis_name == 'ALL VIEW':
                axis_widget = self.all_view_host_main
            elif axis_name in ('U axis', 'V axis', 'W axis'):
                letter = axis_name[0].lower()  # 'u' / 'v' / 'w'
                axis_widget = self._create_axis_tab(letter, joint_type='rotation')
            elif axis_name in ('X axis', 'Y axis', 'Z axis'):
                letter = axis_name[0].lower()  # 'x' / 'y' / 'z'
                axis_widget = self._create_axis_tab(letter, joint_type='translation')
            else:
                axis_widget = self._create_simple_axis_tab(axis_name)
            self.top_subtabs.addTab(axis_widget, axis_name)

        # 初期マウント先はトップの ALL VIEW ホスト
        self._mount_all_view_into(self.all_view_host_main)

        # トップ階層タブの切替で、ALL VIEW のマウント先と共有カメラを反映
        self.top_subtabs.currentChanged.connect(self._on_top_subtab_changed)

        layout.addWidget(self.top_subtabs)
        self.setLayout(layout)

    # 新方式（base STL への RANSAC→ICP フィッティングで重ね合わせ）を使う軸。
    # U〜Z の全軸で有効。FE タブ（base 無し）は常に旧方式。
    BASE_FIT_AXES = {'u', 'v', 'w', 'x', 'y', 'z'}

    def _uses_base_fitting(self, axis_letter) -> bool:
        """指定の軸が新方式（base へのフィッティング）を使うか。"""
        if getattr(self, 'fe_mode', False):
            return False
        return axis_letter in type(self).BASE_FIT_AXES

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

    def _detect_jp_font(self):
        """日本語フォントのパスを検出する。優先順位: メイリオ > 游ゴシック > MS ゴシック。"""
        candidates = [
            r'C:\Windows\Fonts\meiryo.ttc',
            r'C:\Windows\Fonts\meiryob.ttc',
            r'C:\Windows\Fonts\YuGothM.ttc',
            r'C:\Windows\Fonts\YuGothR.ttc',
            r'C:\Windows\Fonts\msgothic.ttc',
        ]
        for path in candidates:
            try:
                if os.path.isfile(path):
                    return path
            except Exception:
                continue
        return None

    def _apply_jp_font(self, ret):
        """add_point_labels / add_text 等の戻り値（actor）に日本語フォントを適用。"""
        font_path = getattr(self, '_jp_font_path', None)
        if not font_path:
            return ret
        actor = ret
        if isinstance(ret, (tuple, list)) and ret:
            actor = ret[-1]
        if actor is None:
            return ret

        # 可能なテキストプロパティを集める（add_point_labels / add_text で経路が異なる）
        text_props = []
        try:
            mp = actor.GetMapper()
            if mp is not None:
                tp = mp.GetLabelTextProperty()
                if tp is not None:
                    text_props.append(tp)
        except Exception:
            pass
        try:
            tp = actor.GetTextProperty()
            if tp is not None:
                text_props.append(tp)
        except Exception:
            pass

        for tp in text_props:
            try:
                tp.SetFontFamily(4)  # 4 = VTK_FONT_FILE
                tp.SetFontFile(font_path)
            except Exception:
                pass
        return ret

    def _setup_plotter_jp_fonts(self, plotter):
        """plotter インスタンスの add_point_labels / add_text をラップして、
        返ってきた actor へ自動で日本語フォントを適用する。
        各 QtInteractor 生成直後に呼ぶ。1 度だけラップされる。"""
        if plotter is None:
            return
        if not getattr(self, '_jp_font_path', None):
            return
        if getattr(plotter, '_jp_font_wrapped', False):
            return

        orig_apl = plotter.add_point_labels
        orig_at = plotter.add_text

        def _wrapped_add_point_labels(*args, **kwargs):
            ret = orig_apl(*args, **kwargs)
            try:
                self._apply_jp_font(ret)
            except Exception:
                pass
            return ret

        def _wrapped_add_text(*args, **kwargs):
            ret = orig_at(*args, **kwargs)
            try:
                self._apply_jp_font(ret)
            except Exception:
                pass
            return ret

        try:
            plotter.add_point_labels = _wrapped_add_point_labels
            plotter.add_text = _wrapped_add_text
            plotter._jp_font_wrapped = True
        except Exception:
            pass

    def _configure_lights(self, plotter, c_world=None, scene_scale=None):
        if plotter is None or not HAS_PYVISTA:
            return
        plotter.remove_all_lights()

        if not get_lighting_enabled():
            try:
                plotter.enable_eye_dome_lighting()
            except Exception:
                pass
            return

        try:
            plotter.disable_eye_dome_lighting()
        except Exception:
            pass

        if c_world is None or scene_scale is None:
            key = pv.Light(position=(3.0, 2.0, 2.5), focal_point=(0.0, 0.0, 0.0), color='white', intensity=1.0)
            fill = pv.Light(position=(-2.0, -1.5, 1.5), focal_point=(0.0, 0.0, 0.0), color='#cfd7ff', intensity=0.45)
            rim = pv.Light(position=(-1.5, 2.5, -2.0), focal_point=(0.0, 0.0, 0.0), color='#fff1d6', intensity=0.35)

            plotter.add_light(key)
            plotter.add_light(fill)
            plotter.add_light(rim)
            return

        origin = np.asarray(c_world.get('origin', [0.0, 0.0, 0.0]), dtype=float)
        ex = np.asarray(c_world.get('ex', [1.0, 0.0, 0.0]), dtype=float)
        ez = np.asarray(c_world.get('ez', [0.0, 0.0, 1.0]), dtype=float)
        ex_norm = float(np.linalg.norm(ex)) or 1.0
        ez_norm = float(np.linalg.norm(ez)) or 1.0
        ex = ex / ex_norm
        ez = ez / ez_norm

        dist = max(float(scene_scale) * 1.2, 1.0)
        key_pos = origin + ex * dist
        fill_pos = origin - ex * dist * 0.6
        rim_pos = origin + ez * dist * 0.6

        key = pv.Light(position=tuple(key_pos), focal_point=tuple(origin), color='white', intensity=1.0)
        fill = pv.Light(position=tuple(fill_pos), focal_point=tuple(origin), color='#cfd7ff', intensity=0.35)
        rim = pv.Light(position=tuple(rim_pos), focal_point=tuple(origin), color='#fff1d6', intensity=0.25)

        plotter.add_light(key)
        plotter.add_light(fill)
        plotter.add_light(rim)

    def _world_light_spec(self):
        """C_world 座標系での共通照明仕様。base STL と C_world が揃っていれば返す。

        定義: base STL の重心を焦点とし、C_world の +X（赤ベクトル＝YZ平面に垂直）方向に
        キーライトを置く。これにより、どのタブでも同じ照明位置になる。
        返り値: dict {focal, key, fill, rim}（すべて C_world 座標の点）/ None。
        """
        base = getattr(self, 'all_view_widget', None)
        if base is None:
            return None
        base_T = self._base_T_world()  # base STL 座標 → C_world 座標
        base_mesh = getattr(base, 'current_mesh', None)
        if base_T is None or base_mesh is None:
            return None
        try:
            c = np.asarray(base_mesh.center, dtype=float)  # base STL 座標での重心(bbox中心)
            b = base_mesh.bounds
            diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])) or 100.0
        except Exception:
            return None
        R = base_T[:3, :3]
        t = base_T[:3, 3]
        focal_w = R @ c + t  # 重心を C_world 座標へ
        dist = max(diag * 1.2, 1.0)
        x = np.array([1.0, 0.0, 0.0])  # C_world +X（赤, YZ平面に垂直）
        z = np.array([0.0, 0.0, 1.0])
        return {
            'focal': focal_w,
            'key': focal_w + x * dist,
            'fill': focal_w - x * dist * 0.6,
            'rim': focal_w + z * dist * 0.6,
        }

    def _configure_lights_for_widget(self, widget, plotter=None):
        """ウィジェット（の表示座標系）に、共通の世界照明を変換して適用する。
        base/C_world が未準備なら従来の固定照明にフォールバック。"""
        if plotter is None:
            plotter = getattr(widget, 'plotter', None)
        if plotter is None or not HAS_PYVISTA:
            return
        plotter.remove_all_lights()
        if not get_lighting_enabled():
            try:
                plotter.enable_eye_dome_lighting()
            except Exception:
                pass
            return
        try:
            plotter.disable_eye_dome_lighting()
        except Exception:
            pass

        spec = self._world_light_spec()
        in_world = getattr(widget, 'display_in_world_frame', False)
        cw = None if in_world else self._effective_c_world_for_widget(widget)
        # 世界照明が使えない / 表示座標への変換ができない → 固定照明
        if spec is None or (not in_world and cw is None):
            key = pv.Light(position=(3.0, 2.0, 2.5), focal_point=(0.0, 0.0, 0.0), color='white', intensity=1.0)
            fill = pv.Light(position=(-2.0, -1.5, 1.5), focal_point=(0.0, 0.0, 0.0), color='#cfd7ff', intensity=0.45)
            rim = pv.Light(position=(-1.5, 2.5, -2.0), focal_point=(0.0, 0.0, 0.0), color='#fff1d6', intensity=0.35)
            plotter.add_light(key)
            plotter.add_light(fill)
            plotter.add_light(rim)
            return

        if cw is None:
            R_w = np.eye(3)
            O_w = np.zeros(3)
        else:
            R_w = np.column_stack([cw['ex'], cw['ey'], cw['ez']])
            O_w = np.asarray(cw['origin'], dtype=float)

        def to_disp(p_world):
            return O_w + R_w @ np.asarray(p_world, dtype=float)

        focal = to_disp(spec['focal'])
        key = pv.Light(position=tuple(to_disp(spec['key'])), focal_point=tuple(focal), color='white', intensity=1.0)
        fill = pv.Light(position=tuple(to_disp(spec['fill'])), focal_point=tuple(focal), color='#cfd7ff', intensity=0.35)
        rim = pv.Light(position=tuple(to_disp(spec['rim'])), focal_point=tuple(focal), color='#fff1d6', intensity=0.25)
        plotter.add_light(key)
        plotter.add_light(fill)
        plotter.add_light(rim)

    def _add_lighting_toggle(self, left_layout, widget):
        cb = QCheckBox('光源を有効化')
        cb.setChecked(get_lighting_enabled())
        cb.toggled.connect(self._on_lighting_toggled)
        left_layout.addWidget(cb)
        self.lighting_checkboxes.append(cb)
        widget.lighting_checkbox = cb

    def _on_lighting_toggled(self, checked: bool):
        set_lighting_enabled(bool(checked))

    def _on_global_lighting_changed(self, enabled: bool):
        for cb in list(self.lighting_checkboxes):
            if cb is None:
                continue
            cb.blockSignals(True)
            cb.setChecked(bool(enabled))
            cb.blockSignals(False)
        self._refresh_all_lighting()

    def _refresh_all_lighting(self):
        for widget in list(self.visual_widgets):
            plotter = getattr(widget, 'plotter', None)
            if plotter is None:
                continue
            view_kind = getattr(widget, 'view_kind', '')
            if view_kind == 'all_view':
                if getattr(widget, 'current_mesh', None) is not None:
                    self._render_all_view(widget, reset_view=False)
                else:
                    self._reset_plotter_placeholder(widget.plotter, 'STL(base) を読み込んでください')
            elif view_kind == 'posture':
                if getattr(widget, 'current_mesh', None) is not None:
                    self._render_posture1_plotter(widget, reset_view=False)
                else:
                    self._reset_plotter_placeholder(widget.plotter, 'STLを読み込んでください')
            elif view_kind == 'motion':
                self._render_motion_view(widget)
    
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

        clear_stl_btn = QPushButton('読み込んだSTLを消去')
        clear_stl_btn.setEnabled(False)
        left_layout.addWidget(clear_stl_btn)

        # === 固定部(C_world)領域 STL（固定部フィットの基準 target）===
        widget.region_mesh = None
        widget.region_stl_path = None
        load_region_btn = QPushButton('固定部(C_world)領域STLを読み込む')
        clear_region_btn = QPushButton('固定部領域STLを消去')
        clear_region_btn.setEnabled(False)
        widget.load_region_btn = load_region_btn
        widget.clear_region_btn = clear_region_btn
        left_layout.addWidget(load_region_btn)
        left_layout.addWidget(clear_region_btn)

        self._add_lighting_toggle(left_layout, widget)

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

        # === 視点記録ボタン ===
        view_btn_row = QHBoxLayout()
        record_view_btn = QPushButton('視点を記録')
        restore_view_btn = QPushButton('記録視点に戻す')
        view_btn_row.addWidget(record_view_btn)
        view_btn_row.addWidget(restore_view_btn)
        left_layout.addLayout(view_btn_row)
        record_view_btn.clicked.connect(lambda: self._record_view(widget))
        restore_view_btn.clicked.connect(lambda: self._restore_recorded_view(widget))

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
            self._setup_plotter_jp_fonts(plotter)
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
        widget.display_in_world_frame = False  # base STL は自前の座標系で表示
        widget.view_kind = 'all_view'
        self._attach_camera_observer(widget)

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
        widget.clear_stl_btn = clear_stl_btn
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
            widget.clear_stl_btn.setEnabled(False)

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
            widget.clear_stl_btn.setEnabled(True)
            self._save_posture_cache(widget)
            # base STL を各軸の base 姿勢へ反映し、同期照明も更新
            self._propagate_base_mesh_to_poses()
            self._refresh_all_lighting()

        def _on_load_error(msg: str):
            widget.log_view.append(msg)
            widget.load_btn.setEnabled(True)
            widget.clear_stl_btn.setEnabled(widget.current_mesh is not None)

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
                        left_clicking=True, right_clicking=True, show_point=False, pickable_window=False,
                    )
                except Exception:
                    pass
            self._render_all_view(widget, reset_view=False)
        widget._on_plane_subtab_changed = _on_plane_subtab_changed
        plane_subtabs.currentChanged.connect(_on_plane_subtab_changed)

        # === ボタン接続 ===
        load_btn.clicked.connect(lambda: self._open_posture_file(widget))
        clear_stl_btn.clicked.connect(lambda: self._clear_all_view_stl(widget))
        load_region_btn.clicked.connect(lambda: self._open_region_file(widget))
        clear_region_btn.clicked.connect(lambda: self._clear_region_stl(widget))
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

        # 任意領域 STL のキャッシュを復元
        region_path = getattr(widget, 'region_stl_path', None)
        if region_path and os.path.exists(region_path):
            widget.log_view.append(f'任意領域キャッシュ検出: {region_path}')
            self._load_region_stl(widget, region_path, from_cache=True)
        elif region_path:
            widget.log_view.append(f'前回の任意領域STLが見つかりません: {region_path}')
            widget.region_stl_path = None

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
        # C_world ができたので共有カメラを反映
        self._apply_shared_camera_with_render(widget)
        # base C_world が確定 → 全タブの同期照明を更新
        self._refresh_all_lighting()
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

    def _reset_plotter_placeholder(self, plotter, message: str):
        if plotter is None or not HAS_PYVISTA:
            return
        try:
            plotter.disable_picking()
        except Exception:
            pass
        plotter.clear()
        background_color, _model_color = self._load_visual_settings()
        plotter.set_background(background_color, top=self._background_top_color(background_color))
        self._configure_lights(plotter)
        plotter.add_text(message, position='upper_left', font_size=10)
        plotter.render()

    def _clear_all_view_stl(self, widget):
        widget.current_mesh = None
        widget.stl_path = None
        widget.c_world = None
        widget.rotation_axes = None
        widget.parallel_axes = None
        widget.intersection_point = None
        widget.intersection_distances = None
        widget.parallel_pair_angles = None
        widget.check_label.setText('（軸を取り込むと結果がここに表示されます）')

        for plane_widget in widget.plane_widgets:
            plane_widget.current_mesh = None
            plane_widget.points.clear()
            plane_widget.selected_point_index = -1
            plane_widget.point_add_enabled = False
            plane_widget.point_add_btn.setChecked(False)
            plane_widget.point_add_btn.setEnabled(False)
            plane_widget._refresh_point_list()

        self._reset_plotter_placeholder(widget.plotter, 'STL(base) を読み込んでください')

        widget.build_world_btn.setEnabled(False)
        widget.clear_world_btn.setEnabled(False)
        widget.import_rotation_btn.setEnabled(False)
        widget.import_parallel_btn.setEnabled(False)
        widget.clear_stl_btn.setEnabled(False)
        widget.log_view.append('STLを消去しました。')
        self._save_posture_cache(widget)
        # base が消えたので base 姿勢と同期照明も更新
        self._propagate_base_mesh_to_poses()
        self._refresh_all_lighting()

    def _clear_posture_stl(self, posture_widget):
        posture_widget.current_mesh = None
        posture_widget.stl_path = None
        posture_widget.c_axis = None
        posture_widget.c_world = None

        for plane_widget in posture_widget.plane_widgets:
            plane_widget.current_mesh = None
            plane_widget.points.clear()
            plane_widget.selected_point_index = -1
            plane_widget.point_add_enabled = False
            plane_widget.point_add_btn.setChecked(False)
            plane_widget.point_add_btn.setEnabled(False)
            plane_widget._refresh_point_list()

        self._reset_plotter_placeholder(posture_widget.plotter, 'STLを読み込んでください')

        posture_widget.build_axis_btn.setEnabled(False)
        posture_widget.clear_axis_btn.setEnabled(False)
        posture_widget.build_world_btn.setEnabled(False)
        posture_widget.clear_world_btn.setEnabled(False)
        posture_widget.clear_stl_btn.setEnabled(False)
        posture_widget.log_view.append('STLを消去しました。')
        self._save_posture_cache(posture_widget)
        self._update_posture_tab_title(posture_widget)
        self._invalidate_motion_for_posture(posture_widget)

    def _update_posture_tab_title(self, posture_widget):
        """姿勢サブタブのタブ名を「姿勢k（ファイル名）」に更新する。
        STL 未読込なら基本名（姿勢k）に戻す。base 姿勢は対象外。"""
        if getattr(posture_widget, 'is_base_pose', False):
            return
        base_label = getattr(posture_widget, 'posture_tab_label', None)
        if not base_label:
            return
        axis_letter = getattr(posture_widget, 'axis_letter', None)
        ad = self.axis_data.get(axis_letter) if axis_letter else None
        if not ad:
            return
        subtabs = ad.get('subtabs')
        if subtabs is None:
            return
        idx = subtabs.indexOf(posture_widget)
        if idx < 0:
            return
        stl_path = getattr(posture_widget, 'stl_path', None)
        if stl_path:
            name = os.path.splitext(os.path.basename(stl_path))[0]
            subtabs.setTabText(idx, f'{base_label}（{name}）')
        else:
            subtabs.setTabText(idx, base_label)

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

    # ----- 視点共有（C_world 基準） -----
    # 各タブで manual に動かしたカメラを C_world 座標系で保存し、別タブにも引き継ぐ。
    # キャッシュは settings.json の tab2.shared_camera_world に保存。

    def _load_shared_camera_cache(self):
        try:
            settings = load_settings() or {}
            sc = (settings.get(self.settings_top_key) or {}).get('shared_camera_world')
            if isinstance(sc, dict) and all(k in sc for k in ('position', 'focal', 'view_up')):
                self.shared_camera_world = {
                    'position': [float(v) for v in sc['position']],
                    'focal': [float(v) for v in sc['focal']],
                    'view_up': [float(v) for v in sc['view_up']],
                }
        except Exception:
            pass

    def _save_shared_camera_cache(self):
        if self.shared_camera_world is None:
            return
        try:
            settings = load_settings() or {}
            top = settings.setdefault(self.settings_top_key, {})
            top['shared_camera_world'] = {
                'position': [float(v) for v in self.shared_camera_world['position']],
                'focal': [float(v) for v in self.shared_camera_world['focal']],
                'view_up': [float(v) for v in self.shared_camera_world['view_up']],
            }
            save_settings(settings)
        except Exception:
            pass

    def _load_recorded_view_cache(self):
        try:
            settings = load_settings() or {}
            rv = (settings.get(self.settings_top_key) or {}).get('recorded_view_world')
            if isinstance(rv, dict) and all(k in rv for k in ('position', 'focal', 'view_up')):
                self.recorded_view_world = {
                    'position': [float(v) for v in rv['position']],
                    'focal': [float(v) for v in rv['focal']],
                    'view_up': [float(v) for v in rv['view_up']],
                }
        except Exception:
            pass

    def _save_recorded_view_cache(self):
        try:
            settings = load_settings() or {}
            top = settings.setdefault(self.settings_top_key, {})
            if self.recorded_view_world is None:
                top.pop('recorded_view_world', None)
            else:
                top['recorded_view_world'] = {
                    'position': [float(v) for v in self.recorded_view_world['position']],
                    'focal': [float(v) for v in self.recorded_view_world['focal']],
                    'view_up': [float(v) for v in self.recorded_view_world['view_up']],
                }
            save_settings(settings)
        except Exception:
            pass

    # ----- motion-axis (回転軸/並進軸) のキャッシュ -----
    def _capture_motion_source_state(self, posture_widgets):
        """motion-axis のソースとなる c_axis/c_world/fit のスナップショット。"""
        # base の C_world も新方式ではソースの一部
        base_cw = None
        base_widget = getattr(self, 'all_view_widget', None)
        if base_widget is not None:
            base_cw = self._serialize_frame(getattr(base_widget, 'c_world', None))
        state = [{'base_c_world': base_cw}]
        # base 姿勢（C_*-axis_base）もソースに含める
        for p in (['base'] + list(type(self).POSTURE_LABELS)):
            pw = posture_widgets.get(p) if posture_widgets else None
            ca = getattr(pw, 'c_axis', None) if pw else None
            cw = getattr(pw, 'c_world', None) if pw else None
            ft = getattr(pw, 'fit_transform', None) if pw else None
            fl = getattr(pw, 'fit_link_transform', None) if pw else None
            state.append({
                'c_axis': self._serialize_frame(ca),
                'c_world': self._serialize_frame(cw),
                'fit_transform': (np.asarray(ft, dtype=float).tolist() if ft is not None else None),
                'fit_link_transform': (np.asarray(fl, dtype=float).tolist() if fl is not None else None),
            })
        return state

    def _serialize_motion_axis(self, mot):
        if mot is None:
            return None
        try:
            base_T = mot.get('base_T_world')
            return {
                'direction': [float(v) for v in mot['direction']],
                'point': [float(v) for v in mot['point']],
                'R_local_world': [np.asarray(m, dtype=float).tolist() for m in mot['R_local_world']],
                'O_local_world': [np.asarray(o, dtype=float).tolist() for o in mot['O_local_world']],
                'T_stl_to_world': [np.asarray(t, dtype=float).tolist() for t in mot['T_stl_to_world']],
                'use_base_fit': bool(mot.get('use_base_fit', False)),
                'base_T_world': (np.asarray(base_T, dtype=float).tolist() if base_T is not None else None),
                'valid_postures': list(mot.get('valid_postures') or []),
            }
        except Exception:
            return None

    def _deserialize_motion_axis(self, d):
        if not isinstance(d, dict):
            return None
        try:
            base_T = d.get('base_T_world')
            return {
                'direction': np.array(d['direction'], dtype=float),
                'point': np.array(d['point'], dtype=float),
                'R_local_world': [np.array(m, dtype=float) for m in d['R_local_world']],
                'O_local_world': [np.array(o, dtype=float) for o in d['O_local_world']],
                'T_stl_to_world': [np.array(t, dtype=float) for t in d['T_stl_to_world']],
                'use_base_fit': bool(d.get('use_base_fit', False)),
                'base_T_world': (np.array(base_T, dtype=float) if base_T is not None else None),
                'valid_postures': list(d.get('valid_postures') or []),
            }
        except Exception:
            return None

    def _save_motion_axis_cache(self, motion_widget):
        axis_letter = getattr(motion_widget, 'axis_letter', None)
        if not axis_letter:
            return
        try:
            settings = load_settings() or {}
            top = settings.setdefault(self.settings_top_key, {})
            key = f'{axis_letter}_motion'
            mot = getattr(motion_widget, 'motion_axis', None)
            if mot is None:
                top.pop(key, None)
            else:
                top[key] = {
                    'motion_axis': self._serialize_motion_axis(mot),
                    'source_state': self._capture_motion_source_state(motion_widget.posture_widgets),
                    'log_text': motion_widget.log_view.toPlainText(),
                    'joint_type': getattr(motion_widget, 'joint_type', 'rotation'),
                }
            save_settings(settings)
        except Exception:
            pass

    def _load_motion_axis_cache(self, motion_widget):
        axis_letter = getattr(motion_widget, 'axis_letter', None)
        if not axis_letter:
            return
        try:
            settings = load_settings() or {}
            entry = (settings.get(self.settings_top_key) or {}).get(f'{axis_letter}_motion') or {}
        except Exception:
            entry = {}
        if not entry:
            motion_widget.motion_axis = None
            motion_widget.needs_update = False
            return
        mot = self._deserialize_motion_axis(entry.get('motion_axis'))
        motion_widget.motion_axis = mot
        log_text = entry.get('log_text') or ''
        if log_text:
            try:
                motion_widget.log_view.setPlainText(log_text)
            except Exception:
                pass
        # 現在のソース状態と比較し needs_update を判定
        try:
            cached_state = entry.get('source_state') or []
            current_state = self._capture_motion_source_state(motion_widget.posture_widgets)
            motion_widget.needs_update = (cached_state != current_state)
        except Exception:
            motion_widget.needs_update = True

    def _invalidate_motion_for_posture(self, posture_widget):
        """指定の posture 側で c_axis または c_world が変わった場合に呼ぶ。
        対応する motion_widget を needs_update にし、再描画。"""
        axis_letter = getattr(posture_widget, 'axis_letter', None)
        if not axis_letter or axis_letter not in self.axis_data:
            return
        mw = self.axis_data[axis_letter].get('motion_widget')
        if mw is None:
            return
        mw.needs_update = True
        try:
            self._save_motion_axis_cache(mw)
        except Exception:
            pass
        try:
            self._render_motion_view(mw)
        except Exception:
            pass

    def _maybe_render_motion_after_mesh_load(self, posture_widget):
        """姿勢 STL がロード完了したタイミングで、3 姿勢が揃っていれば motion-axis を再描画。"""
        axis_letter = getattr(posture_widget, 'axis_letter', None)
        if not axis_letter or axis_letter not in self.axis_data:
            return
        mw = self.axis_data[axis_letter].get('motion_widget')
        if mw is None:
            return
        all_loaded = all(
            getattr(pw, 'current_mesh', None) is not None
            for pw in mw.posture_widgets.values()
        )
        if not all_loaded:
            return
        try:
            self._render_motion_view(mw)
        except Exception:
            pass

    def _record_view(self, widget):
        """現在のカメラを C_world 座標系で記録（手動スナップショット）。"""
        plotter = getattr(widget, 'plotter', None)
        log = getattr(widget, 'log_view', None)
        if plotter is None or not HAS_PYVISTA:
            return
        in_world = getattr(widget, 'display_in_world_frame', False)
        c_world = None if in_world else self._effective_c_world_for_widget(widget)
        if not in_world and c_world is None:
            if log is not None:
                log.append('視点を記録できません（C_world が未生成のため、座標変換できません）。')
            return
        try:
            cam = plotter.camera
            pos = list(cam.position)
            focal = list(cam.focal_point)
            up = list(cam.up)
        except Exception:
            return
        try:
            w_pos, w_focal, w_up = self._display_to_world_cam(pos, focal, up, c_world)
        except Exception:
            return
        self.recorded_view_world = {
            'position': w_pos,
            'focal': w_focal,
            'view_up': w_up,
        }
        self._save_recorded_view_cache()
        if log is not None:
            log.append(
                f'視点を記録しました（C_world 座標系）: '
                f'pos=({w_pos[0]:+.3f}, {w_pos[1]:+.3f}, {w_pos[2]:+.3f}), '
                f'up=({w_up[0]:+.3f}, {w_up[1]:+.3f}, {w_up[2]:+.3f})'
            )

    def _restore_recorded_view(self, widget):
        """記録された視点に戻し、共有カメラも同値で上書きしてタブ間に伝搬。"""
        plotter = getattr(widget, 'plotter', None)
        log = getattr(widget, 'log_view', None)
        if plotter is None or not HAS_PYVISTA:
            return
        if self.recorded_view_world is None:
            if log is not None:
                log.append('記録された視点がありません。先に「視点を記録」してください。')
            return
        in_world = getattr(widget, 'display_in_world_frame', False)
        c_world = None if in_world else self._effective_c_world_for_widget(widget)
        if not in_world and c_world is None:
            if log is not None:
                log.append('視点を復元できません（C_world が未生成のため、座標変換できません）。')
            return
        try:
            rv = self.recorded_view_world
            d_pos, d_focal, d_up = self._world_to_display_cam(
                rv['position'], rv['focal'], rv['view_up'], c_world,
            )
            plotter.camera_position = [tuple(d_pos), tuple(d_focal), tuple(d_up)]
            try:
                plotter.reset_camera_clipping_range()
            except Exception:
                pass
            plotter.render()
            # 復元した視点を共有カメラへ反映 → 他タブにも引き継がれる
            self.shared_camera_world = dict(self.recorded_view_world)
            self._save_shared_camera_cache()
            if log is not None:
                log.append('記録された視点に戻しました。')
        except Exception:
            pass

    def _base_T_world(self):
        """base STL 座標 → C_world 座標 の 4x4（ALL VIEW の C_world から）。未生成なら None。"""
        base = getattr(self, 'all_view_widget', None)
        cw = getattr(base, 'c_world', None) if base is not None else None
        if cw is None:
            return None
        R_w = np.column_stack([cw['ex'], cw['ey'], cw['ez']])
        O_w = np.asarray(cw['origin'], dtype=float)
        R_w_T = R_w.T
        T = np.eye(4)
        T[:3, :3] = R_w_T
        T[:3, 3] = -R_w_T @ O_w
        return T

    def _posture_stl_to_world(self, widget):
        """新方式 posture: STL 座標 → C_world 座標 の 4x4。
        = base_T_world @ T_fit。base C_world か fit が無ければ None。"""
        base_T = self._base_T_world()
        if base_T is None:
            return None
        T_fit = getattr(widget, 'fit_transform', None)
        if T_fit is None:
            return None
        return base_T @ np.asarray(T_fit, dtype=float)

    @staticmethod
    def _c_world_from_T_stl_to_world(T):
        """STL→World 変換 (4x4) から、カメラ変換用の c_world 風 dict を作る。
        既存の _display_to_world_cam は p_world = R_w^T (p_stl - O_w) を使うので、
        R_sw = R_w^T, t_sw = -R_w^T O_w を満たす ex/ey/ez/origin を逆算する。"""
        T = np.asarray(T, dtype=float)
        R_sw = T[:3, :3]
        t_sw = T[:3, 3]
        R_w = R_sw.T  # 列が C_world 各軸（STL 座標表現）
        O_w = -R_sw.T @ t_sw
        return {
            'ex': R_w[:, 0].copy(),
            'ey': R_w[:, 1].copy(),
            'ez': R_w[:, 2].copy(),
            'origin': O_w.copy(),
        }

    def _effective_c_world_for_widget(self, widget):
        """カメラ共有用に、このウィジェットの「C_world を STL 座標で表した dict」を返す。
        - world 系で描画するウィジェット（motion 等）→ None（変換不要）
        - 新方式 posture → base_T_world @ T_fit から逆算
        - それ以外（ALL VIEW / 旧方式 posture）→ widget.c_world をそのまま
        """
        if getattr(widget, 'display_in_world_frame', False):
            return None
        if getattr(widget, 'use_base_fit', False) and getattr(widget, 'view_kind', None) == 'posture':
            T = self._posture_stl_to_world(widget)
            if T is None:
                return None
            return self._c_world_from_T_stl_to_world(T)
        return getattr(widget, 'c_world', None)

    def _display_to_world_cam(self, position, focal, view_up, c_world):
        pos = np.asarray(position, dtype=float)
        foc = np.asarray(focal, dtype=float)
        up = np.asarray(view_up, dtype=float)
        if c_world is None:
            return pos.tolist(), foc.tolist(), up.tolist()
        R_w = np.column_stack([c_world['ex'], c_world['ey'], c_world['ez']])
        O_w = np.asarray(c_world['origin'], dtype=float)
        R_w_T = R_w.T
        return (
            (R_w_T @ (pos - O_w)).tolist(),
            (R_w_T @ (foc - O_w)).tolist(),
            (R_w_T @ up).tolist(),
        )

    def _world_to_display_cam(self, position, focal, view_up, c_world):
        pos = np.asarray(position, dtype=float)
        foc = np.asarray(focal, dtype=float)
        up = np.asarray(view_up, dtype=float)
        if c_world is None:
            return pos.tolist(), foc.tolist(), up.tolist()
        R_w = np.column_stack([c_world['ex'], c_world['ey'], c_world['ez']])
        O_w = np.asarray(c_world['origin'], dtype=float)
        return (
            (O_w + R_w @ pos).tolist(),
            (O_w + R_w @ foc).tolist(),
            (R_w @ up).tolist(),
        )

    def _on_camera_interaction_end(self, widget):
        plotter = getattr(widget, 'plotter', None)
        if plotter is None or not HAS_PYVISTA:
            return
        in_world = getattr(widget, 'display_in_world_frame', False)
        c_world = None if in_world else self._effective_c_world_for_widget(widget)
        if not in_world and c_world is None:
            return  # 変換に C_world が要るが未生成
        try:
            cam = plotter.camera
            pos = list(cam.position)
            focal = list(cam.focal_point)
            up = list(cam.up)
        except Exception:
            return
        try:
            w_pos, w_focal, w_up = self._display_to_world_cam(pos, focal, up, c_world)
        except Exception:
            return
        self.shared_camera_world = {
            'position': w_pos,
            'focal': w_focal,
            'view_up': w_up,
        }
        self._save_shared_camera_cache()

    def _apply_shared_camera_to_plotter(self, plotter, widget):
        """shared_camera_world を、widget の座標系を介して指定 plotter に適用。
        plotter は widget.plotter とは別でもよい（ミラー用）。成功すれば True。"""
        if self.shared_camera_world is None:
            return False
        if plotter is None or not HAS_PYVISTA:
            return False
        in_world = getattr(widget, 'display_in_world_frame', False)
        c_world = None if in_world else self._effective_c_world_for_widget(widget)
        if not in_world and c_world is None:
            return False
        try:
            sc = self.shared_camera_world
            d_pos, d_focal, d_up = self._world_to_display_cam(
                sc['position'], sc['focal'], sc['view_up'], c_world,
            )
            plotter.camera_position = [tuple(d_pos), tuple(d_focal), tuple(d_up)]
            try:
                plotter.reset_camera_clipping_range()
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _set_camera_from_shared(self, widget):
        """shared_camera_world をこのウィジェットの表示座標へ変換して適用。
        render() は呼ばない。成功すれば True。"""
        return self._apply_shared_camera_to_plotter(getattr(widget, 'plotter', None), widget)

    def _apply_shared_camera_with_render(self, widget):
        if self._set_camera_from_shared(widget):
            try:
                widget.plotter.render()
            except Exception:
                pass

    def _mount_all_view_into(self, host):
        """ALL VIEW 本体ウィジェットを host のレイアウトへ付け替える（reparent）。"""
        av = getattr(self, 'all_view_widget', None)
        if av is None or host is None:
            return
        lay = host.layout()
        if lay is None:
            lay = QVBoxLayout(host)
            lay.setContentsMargins(0, 0, 0, 0)
        if av.parentWidget() is host:
            return
        old = av.parentWidget()
        if old is not None and old.layout() is not None:
            old.layout().removeWidget(av)
        lay.addWidget(av)
        av.show()
        # reparent 後は GL サーフェスを更新
        plotter = getattr(av, 'plotter', None)
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass

    def _refresh_all_view_mount(self):
        """現在表示中のタブに応じて ALL VIEW 本体のマウント先を決める。"""
        try:
            idx = self.top_subtabs.currentIndex()
            tab_name = self.top_subtabs.tabText(idx)
        except Exception:
            return
        target = self.all_view_host_main
        if tab_name in ('U axis', 'V axis', 'W axis', 'X axis', 'Y axis', 'Z axis'):
            ad = self.axis_data.get(tab_name[0].lower())
            if ad:
                sub = ad['subtabs']
                host = ad.get('mirror_host')
                # 現在のサブタブが ALL VIEW ホストなら、そこへマウント
                if host is not None and sub.widget(sub.currentIndex()) is host:
                    target = host
        self._mount_all_view_into(target)

    def _active_widget_in_axis(self, axis_letter):
        ad = self.axis_data.get(axis_letter)
        if not ad:
            return None
        subtabs = ad['subtabs']
        idx = subtabs.currentIndex()
        w = None
        try:
            w = subtabs.widget(idx)
        except Exception:
            return None
        # ALL VIEW ホスト（本体を差し込むだけの器）なら、本体ウィジェットを返す
        if w is ad.get('mirror_host'):
            return self.all_view_widget
        return w

    def _on_axis_subtab_changed(self, axis_letter):
        self._refresh_all_view_mount()
        widget = self._active_widget_in_axis(axis_letter)
        if widget is not None:
            self._apply_shared_camera_with_render(widget)

    def _on_top_subtab_changed(self, index):
        self._refresh_all_view_mount()
        try:
            tab_name = self.top_subtabs.tabText(index)
        except Exception:
            return
        if tab_name == 'ALL VIEW':
            if self.all_view_widget is not None:
                self._apply_shared_camera_with_render(self.all_view_widget)
        elif tab_name in ('U axis', 'V axis', 'W axis', 'X axis', 'Y axis', 'Z axis'):
            letter = tab_name[0].lower()
            widget = self._active_widget_in_axis(letter)
            if widget is not None:
                self._apply_shared_camera_with_render(widget)

    def _attach_camera_observer(self, widget):
        plotter = getattr(widget, 'plotter', None)
        if plotter is None or not HAS_PYVISTA:
            return
        cb = lambda *_a, w=widget: self._on_camera_interaction_end(w)
        try:
            plotter.iren.add_observer('EndInteractionEvent', cb)
            return
        except Exception:
            pass
        try:
            plotter.iren.AddObserver('EndInteractionEvent', cb)
            return
        except Exception:
            pass
        try:
            plotter.interactor.AddObserver('EndInteractionEvent', cb)
        except Exception:
            pass

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
        b = widget.current_mesh.bounds
        diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])) or 100.0
        self._configure_lights_for_widget(widget, plotter=plotter)
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
            bounds = widget.current_mesh.bounds
            if not self._apply_shared_camera_to_plotter(plotter, widget):
                plotter.reset_camera(bounds=bounds)
        elif camera_position is not None:
            try:
                plotter.camera_position = camera_position
            except Exception:
                pass

        if active_plane is not None and active_plane.point_add_enabled:
            try:
                plotter.enable_surface_point_picking(
                    callback=lambda point, *_args: self._on_plane_surface_point_picked(active_plane, point),
                    left_clicking=True, right_clicking=True, show_point=False, pickable_window=False,
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
        postures = list(type(self).POSTURE_SPECS)
        posture_widgets = {}

        # 姿勢1 の左に、ALL VIEW 本体を差し込むためのホスト（空の器）を配置。
        # FE タブには base が無いので作らない。
        mirror_host = None
        base_widget = None
        if not getattr(self, 'fe_mode', False):
            mirror_host = QWidget()
            _mh = QVBoxLayout(mirror_host)
            _mh.setContentsMargins(0, 0, 0, 0)
            posture_subtabs.addTab(mirror_host, 'ALL VIEW')

            # ALL VIEW の次に「base」姿勢（base STL 上にその軸用の C_*-axis_base を作る）
            base_widget = self._create_posture_view_widget(
                'base', '', 'base', axis_letter, is_base_pose=True,
            )
            posture_widgets['base'] = base_widget
            posture_subtabs.addTab(base_widget, 'base')

        for posture_label, example_text, posture_key in postures:
            posture_widget = self._create_posture_view_widget(
                posture_label, example_text, posture_key, axis_letter
            )
            posture_widgets[posture_label] = posture_widget
            posture_subtabs.addTab(posture_widget, posture_label)

        # 末尾に「X軸回転軸 / 並進軸」タブを追加（姿勢表示切替は全軸で有効化）
        motion_widget = self._create_motion_axis_tab(
            axis_letter, posture_widgets, joint_type=joint_type,
            enable_posture_controls=True,
        )
        tab_label = f'{axis_letter.upper()}軸回転軸' if joint_type == 'rotation' else f'{axis_letter.upper()}軸並進軸'
        posture_subtabs.addTab(motion_widget, tab_label)

        main_layout.addWidget(posture_subtabs, 1)

        self.axis_data[axis_letter] = {
            'subtabs': posture_subtabs,
            'posture_widgets': posture_widgets,
            'motion_widget': motion_widget,
            'mirror_host': mirror_host,
            'base_widget': base_widget,
            'joint_type': joint_type,
        }

        # 軸内のサブタブ（姿勢1/2/3 / motion-axis）切替時に共有カメラを反映
        posture_subtabs.currentChanged.connect(
            lambda _idx, l=axis_letter: self._on_axis_subtab_changed(l)
        )

        # motion-axis のキャッシュを復元（STL は async で読み込まれるため、
        # 描画は _on_mesh_loaded から再トリガーされる）
        self._load_motion_axis_cache(motion_widget)

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

        self._add_lighting_toggle(left_layout, widget)

        # === 視点記録ボタン ===
        view_btn_row = QHBoxLayout()
        record_view_btn = QPushButton('視点を記録')
        restore_view_btn = QPushButton('記録視点に戻す')
        view_btn_row.addWidget(record_view_btn)
        view_btn_row.addWidget(restore_view_btn)
        left_layout.addLayout(view_btn_row)
        record_view_btn.clicked.connect(lambda: self._record_view(widget))
        restore_view_btn.clicked.connect(lambda: self._restore_recorded_view(widget))

        # 既定の表示名（Ver.2 でも同じ既定値を使うが UI からは触れない）
        widget.cworld_display_name = 'C_world'
        widget.show_captions = True  # Ver.2 では常に True、UI からは触らない

        # FE 専用: ラベル名（C_world / motion 軸）の自由入力 + キャプション表示トグル
        if getattr(self, 'fe_mode', False):
            name_group = QGroupBox('ラベル名（このタブのみ）')
            name_layout = QVBoxLayout(name_group)

            # キャプション全体の表示/非表示
            show_caption_cb = QCheckBox('キャプションを表示')
            show_caption_cb.setChecked(True)
            name_layout.addWidget(show_caption_cb)
            widget.show_caption_cb = show_caption_cb

            def _on_show_caption_toggled(checked, w=widget):
                w.show_captions = bool(checked)
                try:
                    self._render_motion_view(w)
                except Exception:
                    pass
            show_caption_cb.toggled.connect(_on_show_caption_toggled)

            cw_row = QHBoxLayout()
            cw_row.addWidget(QLabel('C_world 名:'))
            cworld_edit = QLineEdit('C_world')
            cworld_edit.setPlaceholderText('例: C_world / W_FE / ...')
            cw_row.addWidget(cworld_edit, 1)
            name_layout.addLayout(cw_row)

            mn_row = QHBoxLayout()
            mn_row.addWidget(QLabel(f'{widget.motion_label_jp}名:'))
            motion_edit = QLineEdit(widget.motion_name)
            motion_edit.setPlaceholderText('例: U_rotation-axis / FE_rot / ...')
            mn_row.addWidget(motion_edit, 1)
            name_layout.addLayout(mn_row)

            left_layout.addWidget(name_group)
            widget.cworld_name_edit = cworld_edit
            widget.motion_name_edit = motion_edit

            def _on_cworld_name_changed(text, w=widget):
                w.cworld_display_name = (text.strip() if text else '') or 'C_world'
                try:
                    self._render_motion_view(w)
                except Exception:
                    pass
            cworld_edit.textChanged.connect(_on_cworld_name_changed)

            def _on_motion_name_changed(text, w=widget):
                stripped = (text or '').strip()
                # 空文字は無視（既存値を保持）
                if stripped:
                    w.motion_name = stripped
                    try:
                        self._render_motion_view(w)
                    except Exception:
                        pass
            motion_edit.textChanged.connect(_on_motion_name_changed)

        opacity_group = QGroupBox('STL 透明度')
        opacity_layout = QVBoxLayout(opacity_group)

        sliders = {}
        for posture_label in type(self).POSTURE_LABELS:
            default_val = 50
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
        _posture_labels = type(self).POSTURE_LABELS
        widget.unify_stl_color = False
        widget.stl_visibility = {lbl: True for lbl in _posture_labels}
        widget.caxis_visibility = {lbl: True for lbl in _posture_labels}
        widget.show_cworld = True
        widget.show_base_stl = True
        widget.base_stl_opacity = 0.35

        if enable_posture_controls:
            posture_view_group = QGroupBox('姿勢変化を分かりやすくする')
            pv_layout = QVBoxLayout(posture_view_group)

            unify_cb = QCheckBox('色を統一')
            unify_cb.setChecked(False)
            pv_layout.addWidget(unify_cb)

            cworld_cb = QCheckBox('C_world を表示')
            cworld_cb.setChecked(True)
            pv_layout.addWidget(cworld_cb)

            # 新方式のみ: base STL の表示トグル
            if self._uses_base_fitting(axis_letter):
                base_stl_cb = QCheckBox('STL(base) を表示')
                base_stl_cb.setChecked(True)
                pv_layout.addWidget(base_stl_cb)
                widget.base_stl_cb = base_stl_cb
                base_stl_cb.toggled.connect(
                    lambda c, w=widget: self._on_posture_view_toggled(w, 'base_stl', None, c)
                )

            stl_cbs = {}
            caxis_cbs = {}
            for i in range(1, len(_posture_labels) + 1):
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
            widget.cworld_cb = cworld_cb
            widget.stl_visibility_cbs = stl_cbs
            widget.caxis_visibility_cbs = caxis_cbs

            unify_cb.toggled.connect(
                lambda c, w=widget: self._on_posture_view_toggled(w, 'unify', None, c)
            )
            cworld_cb.toggled.connect(
                lambda c, w=widget: self._on_posture_view_toggled(w, 'cworld', None, c)
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
            self._setup_plotter_jp_fonts(plotter)
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
        widget.display_in_world_frame = True  # motion-axis タブは C_world 系で表示
        widget.view_kind = 'motion'
        self._attach_camera_observer(widget)
        widget.log_view = log_view
        widget.compute_btn = compute_btn
        widget.sliders = sliders
        widget.motion_axis = None
        widget.needs_update = False
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
        elif kind == 'cworld':
            widget.show_cworld = bool(checked)
        elif kind == 'base_stl':
            widget.show_base_stl = bool(checked)
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
        all_postures = list(type(self).POSTURE_LABELS)
        use_base_fit = self._uses_base_fitting(axis_letter)

        # base 情報（新方式のみ）: 単一の C_world は ALL VIEW(base) で生成したもの
        base_widget = getattr(self, 'all_view_widget', None)
        base_T_world = None      # base STL 座標 → C_world 座標 の 4x4
        if use_base_fit:
            if base_widget is None or getattr(base_widget, 'c_world', None) is None:
                log.append('ALL VIEW で C_world 座標系が未生成です。先に base STL の C_world を生成してください。')
                return
            cw = base_widget.c_world
            R_w = np.column_stack([cw['ex'], cw['ey'], cw['ez']])
            O_w = np.asarray(cw['origin'], dtype=float)
            R_w_T = R_w.T
            base_T_world = np.eye(4)
            base_T_world[:3, :3] = R_w_T
            base_T_world[:3, 3] = -R_w_T @ O_w

        # F_L: base 上に定義した C_*-axis（全姿勢で共有するシードフレーム）。
        # 未生成でも軸の向き・位置には影響しないため、その場合は恒等で代用する。
        R_FL = np.eye(3)
        O_FL = np.zeros(3)
        if use_base_fit:
            base_pose = posture_widgets.get('base')
            F_L = getattr(base_pose, 'c_axis', None) if base_pose else None
            if F_L is not None:
                R_FL = np.column_stack([F_L['ex'], F_L['ey'], F_L['ez']])
                O_FL = np.asarray(F_L['origin'], dtype=float)
            else:
                log.append(f'注意: base 上に {local_prefix}_base が未生成です。表示フレームは原点に置きます（軸の値には影響しません）。')

        # 寛容な入力検証: 有効な姿勢を抽出（新方式では base も 1 個の姿勢として含める）
        # valid 要素: (label, posture_widget, M, T_world)
        #   新方式: M = T_world ∘ inv(T_link)（リンクの剛体移動）, base は M=I
        candidate_labels = list(all_postures)
        if use_base_fit and 'base' in posture_widgets:
            candidate_labels = ['base'] + candidate_labels
        valid = []
        for p in candidate_labels:
            pw = posture_widgets.get(p)
            if pw is None:
                continue
            if getattr(pw, 'current_mesh', None) is None:
                log.append(f'{p}: STL 未読込のためスキップ')
                continue
            if use_base_fit:
                if p == 'base':
                    valid.append((p, pw, np.eye(4), np.eye(4)))
                    continue
                Tw = getattr(pw, 'fit_transform', None)
                Tl = getattr(pw, 'fit_link_transform', None)
                if Tw is None:
                    log.append(f'{p}: 固定部(C_world)フィット未実施のためスキップ')
                    continue
                if Tl is None:
                    log.append(f'{p}: アーム({local_prefix})フィット未実施のためスキップ')
                    continue
                Tw = np.asarray(Tw, dtype=float)
                Tl = np.asarray(Tl, dtype=float)
                try:
                    M = Tw @ np.linalg.inv(Tl)
                except np.linalg.LinAlgError:
                    log.append(f'{p}: アームフィット行列が特異のためスキップ')
                    continue
                valid.append((p, pw, M, Tw))
            else:
                # 旧方式: c_axis + c_world 必須
                if getattr(pw, 'c_axis', None) is None:
                    log.append(f'{p}: {local_prefix} 未生成のためスキップ')
                    continue
                if getattr(pw, 'c_world', None) is None:
                    log.append(f'{p}: C_world 未生成のためスキップ')
                    continue
                valid.append((p, pw, None, None))

        if len(valid) < 2:
            if use_base_fit:
                log.append('有効な姿勢が 2 つ以上ありません。base STL（C_world）と、各姿勢で固定部・アームの2フィットをご準備ください。')
            else:
                log.append('有効な姿勢が 2 つ以上ありません。各姿勢で STL 読込・C_local・C_world をご準備ください。')
            return

        log.append(f'有効な姿勢: {", ".join(v[0] for v in valid)} （計 {len(valid)} 個）')

        # 各姿勢: STL→World 変換、および フレーム(C_*-axis)の World 表現
        R_local_world = []
        O_local_world = []
        T_stl_to_world = []
        if use_base_fit:
            # 新方式: フレーム = M_k ∘ F_L（base座標）→ C_world 表現。
            # STL 重ね描画は固定部フィット T_world で base へ重ねる。
            R_bw = base_T_world[:3, :3]
            t_bw = base_T_world[:3, 3]
            for (_p, pw, M, Tw) in valid:
                R_M = M[:3, :3]
                t_M = M[:3, 3]
                R_frame = R_M @ R_FL
                O_frame = R_M @ O_FL + t_M
                R_local_world.append(R_bw @ R_frame)
                O_local_world.append(R_bw @ O_frame + t_bw)
                T_stl_to_world.append(base_T_world @ Tw)
        else:
            # 旧方式: 各姿勢が自前の C_world を持つ
            for (_p, pw, _M, _Tw) in valid:
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

        N = len(valid)
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        # ログ表示用のラベル（'base' / '姿勢k'）
        pose_labels = [v[0] for v in valid]

        # 全ペアの R_ij と Δ_ij
        R_pairs = []
        theta_list = []
        axis_list = []
        d_list = []
        for (i, j) in pairs:
            R_ij = R_local_world[j] @ R_local_world[i].T
            th, ax = self._rotation_log_axis(R_ij)
            R_pairs.append(R_ij)
            theta_list.append(th)
            axis_list.append(ax)
            d_list.append(O_local_world[j] - O_local_world[i])

        def _deg(v, w):
            nv = np.linalg.norm(v); nw = np.linalg.norm(w)
            if nv < 1e-12 or nw < 1e-12:
                return float('nan')
            return float(np.degrees(np.arccos(np.clip(float(np.dot(v / nv, w / nw)), -1.0, 1.0))))

        if joint_type == 'rotation':
            # 軸方向の符号合わせ（最初の axis を基準）
            ref = axis_list[0].copy()
            axes_signed = [axis_list[0]]
            for k in range(1, len(axis_list)):
                ak = axis_list[k]
                if np.dot(ak, ref) < 0:
                    ak = -ak
                axes_signed.append(ak)

            weights = np.array(theta_list, dtype=float)
            if weights.sum() < 1e-9:
                log.append('全ペアの回転角がほぼゼロです。回転軸を確定できません。')
                return
            avg_dir = np.zeros(3)
            for w, a in zip(weights, axes_signed):
                avg_dir = avg_dir + w * a
            n_avg = np.linalg.norm(avg_dir)
            if n_avg < 1e-9:
                log.append('回転軸の平均方向がゼロです。回転軸を確定できません。')
                return
            avg_dir = avg_dir / n_avg

            # 回転軸の通る点 p を最小二乗で
            I3 = np.eye(3)
            A_blocks = [I3 - R for R in R_pairs]
            b_blocks = [
                O_local_world[j] - R_pairs[k] @ O_local_world[i]
                for k, (i, j) in enumerate(pairs)
            ]
            A = np.vstack(A_blocks)
            b = np.concatenate(b_blocks)
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
                'valid_postures': [v[0] for v in valid],
                'use_base_fit': use_base_fit,
                'base_T_world': base_T_world,
            }

            log.append(f'{motion_name} を計算しました（C_world 座標系上, {N} 姿勢, {len(pairs)} ペア）:')
            for k, (i, j) in enumerate(pairs):
                li, lj = pose_labels[i], pose_labels[j]
                ax = axes_signed[k]
                log.append(
                    f'  {li}→{lj}: 回転角 = {np.degrees(theta_list[k]):.3f}°, '
                    f'軸 = ({ax[0]:+.4f}, {ax[1]:+.4f}, {ax[2]:+.4f})'
                )
            log.append('  --- 統合結果 ---')
            log.append(f'  {motion_name} 方向 = ({avg_dir[0]:+.6f}, {avg_dir[1]:+.6f}, {avg_dir[2]:+.6f})')
            log.append(f'  軸が通る点 (C_world)   = ({p_axis[0]:+.4f}, {p_axis[1]:+.4f}, {p_axis[2]:+.4f})')
            log.append('  --- 平均方向からの各候補の方向ずれ（理想 0°） ---')
            for k, (i, j) in enumerate(pairs):
                li, lj = pose_labels[i], pose_labels[j]
                log.append(f'    ∠({li}→{lj}, 平均) = {_deg(axes_signed[k], avg_dir):.3f}°')
        else:  # translation
            # 並進方向の符号合わせ（最初の Δ を基準）
            ref = d_list[0].copy()
            d_signed = [d_list[0]]
            for k in range(1, len(d_list)):
                dk = d_list[k]
                if np.dot(dk, ref) < 0:
                    dk = -dk
                d_signed.append(dk)

            lens = np.array([np.linalg.norm(d) for d in d_signed], dtype=float)
            if lens.sum() < 1e-9:
                log.append('全ペアの並進距離がほぼゼロです。並進軸を確定できません。')
                return
            avg_dir = np.zeros(3)
            for l, d in zip(lens, d_signed):
                avg_dir = avg_dir + l * d
            n_avg = np.linalg.norm(avg_dir)
            if n_avg < 1e-9:
                log.append('並進軸の平均方向がゼロです。')
                return
            avg_dir = avg_dir / n_avg

            # 軸の通る点: 最初の有効な姿勢の C_local 原点を採用
            p_axis = O_local_world[0]

            widget.motion_axis = {
                'direction': avg_dir,
                'point': p_axis,
                'R_local_world': R_local_world,
                'O_local_world': O_local_world,
                'T_stl_to_world': T_stl_to_world,
                'valid_postures': [v[0] for v in valid],
                'use_base_fit': use_base_fit,
                'base_T_world': base_T_world,
            }

            log.append(f'{motion_name} を計算しました（C_world 座標系上, {N} 姿勢, {len(pairs)} ペア）:')
            log.append('  ※ 姿勢間の回転行列 R（直動関節なら理想 R = I）と平行移動 Δ を表示します。')

            def _fmt_R(R):
                return [
                    f'    [{R[0, 0]:+.6f}, {R[0, 1]:+.6f}, {R[0, 2]:+.6f}]',
                    f'    [{R[1, 0]:+.6f}, {R[1, 1]:+.6f}, {R[1, 2]:+.6f}]',
                    f'    [{R[2, 0]:+.6f}, {R[2, 1]:+.6f}, {R[2, 2]:+.6f}]',
                ]
            for k, (i, j) in enumerate(pairs):
                li, lj = pose_labels[i], pose_labels[j]
                R_ij = R_pairs[k]
                th_ij = theta_list[k]
                dvec = d_list[k]
                tag = f'{li}_{lj}'
                log.append(f'  --- {li}→{lj} ---')
                log.append(f'    R_{tag} (回転角 = {np.degrees(th_ij):.4f}°, 理想 0°):')
                for ln in _fmt_R(R_ij):
                    log.append(ln)
                log.append(
                    f'    Δ_{tag} = '
                    f'({dvec[0]:+.4f}, {dvec[1]:+.4f}, {dvec[2]:+.4f}) [mm], '
                    f'||Δ|| = {np.linalg.norm(dvec):.4f} mm'
                )

            log.append('  --- 統合結果 ---')
            log.append(f'  {motion_name} 方向 = ({avg_dir[0]:+.6f}, {avg_dir[1]:+.6f}, {avg_dir[2]:+.6f})')
            first_label = pose_labels[0]
            log.append(
                f'  軸が通る点 (C_world, {first_label} 原点) '
                f'= ({p_axis[0]:+.4f}, {p_axis[1]:+.4f}, {p_axis[2]:+.4f})'
            )
            log.append('  --- 平均方向からの各候補の方向ずれ（理想 0°） ---')
            for k, (i, j) in enumerate(pairs):
                ni, nj = posture_nums[i], posture_nums[j]
                log.append(f'    ∠(Δ_{ni}_{nj}, 平均) = {_deg(d_signed[k], avg_dir):.4f}°')

        widget.needs_update = False
        self._save_motion_axis_cache(widget)
        self._render_motion_view(widget)

    def _render_motion_view(self, widget):
        plotter = getattr(widget, 'plotter', None)
        if plotter is None or not HAS_PYVISTA:
            return

        plotter.clear()
        background_color, model_color = self._load_visual_settings()
        plotter.set_background(background_color, top=self._background_top_color(background_color))
        self._configure_lights_for_widget(widget, plotter=plotter)
        plotter.hide_axes()

        rot = getattr(widget, 'motion_axis', None)
        needs_update = bool(getattr(widget, 'needs_update', False))
        posture_widgets = getattr(widget, 'posture_widgets', {})
        all_postures = list(type(self).POSTURE_LABELS)
        # rot がある場合は計算に使った姿勢のメッシュが揃っているかで判定
        if rot is not None and isinstance(rot.get('valid_postures'), list) and rot['valid_postures']:
            postures = list(rot['valid_postures'])
        else:
            postures = all_postures
        all_meshes_loaded = all(
            getattr(posture_widgets.get(p), 'current_mesh', None) is not None
            for p in postures
        )
        motion_label_jp = getattr(widget, 'motion_label_jp', '回転軸')

        # 表示できない状態を判定して、ガイダンス文だけ描画して返す
        if needs_update or rot is None or not all_meshes_loaded:
            if needs_update:
                main_msg = '更新が必要です'
                sub_msg = (
                    f'姿勢1〜{len(all_postures)} のいずれかが更新されました。\n'
                    f'「{motion_label_jp}を計算 / 表示更新」をクリックしてください。'
                )
            elif rot is None:
                main_msg = '未計算'
                sub_msg = f'「{motion_label_jp}を計算 / 表示更新」をクリックしてください。'
            else:  # not all_meshes_loaded
                main_msg = 'STL 読み込み中…'
                sub_msg = '計算済姿勢のSTL読み込み完了をお待ちください。'
            try:
                plotter.add_text(main_msg, position='upper_edge', font_size=18,
                                 color='#ffd060', name='motion_msg_main')
            except Exception:
                pass
            try:
                plotter.add_text(sub_msg, position='lower_edge', font_size=11,
                                 color='#cfd6e0', name='motion_msg_sub')
            except Exception:
                pass
            plotter.render()
            return

        axis_letter = getattr(widget, 'axis_letter', 'u')
        motion_name = getattr(widget, 'motion_name', f'{axis_letter.upper()}_motion-axis')
        local_label_prefix = f'C_{axis_letter}-axis'

        T_list = rot['T_stl_to_world']
        # 描画対象の姿勢ラベル列（rot に格納された valid_postures を優先）
        if rot is not None and isinstance(rot.get('valid_postures'), list) and rot['valid_postures']:
            postures = list(rot['valid_postures'])
        else:
            postures = all_postures[: len(T_list)]
        # 姿勢ごとに異なる淡い色で重ねる（5 姿勢まで）
        stl_colors = {
            'base': '#c0c4cc',  # 灰（基準）
            '姿勢1': '#ffc070',  # 橙
            '姿勢2': '#80d0a0',  # 緑
            '姿勢3': '#a0a0ff',  # 紫
            '姿勢4': '#d4d480',  # 黄
            '姿勢5': '#80d4d4',  # シアン
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
            pw = posture_widgets.get(p)
            if pw is None or getattr(pw, 'current_mesh', None) is None:
                continue
            mesh = pw.current_mesh.copy()
            mesh.transform(T, inplace=True)
            b = mesh.bounds
            # 非表示でも bounds は更新（カメラ安定化のため）
            bounds_all = _accumulate_bounds(bounds_all, b)

            # base 姿勢の可視は show_base_stl トグルに連動
            if p == 'base':
                if not getattr(widget, 'show_base_stl', True):
                    continue
            elif not stl_visibility.get(p, True):
                continue

            if p in widget.sliders:
                opacity = widget.sliders[p][0].value() / 100.0
            else:
                opacity = float(getattr(widget, 'base_stl_opacity', 0.35))
            color = unified_model_color if unify_color else stl_colors.get(p, '#c0c4cc')
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

        # C_world の座標軸（原点 = 0, 単位ベクトル）— トグルで非表示にできる
        show_cworld = getattr(widget, 'show_cworld', True)
        show_caps_motion = bool(getattr(widget, 'show_captions', True))
        axis_len_world = max(diag * 0.18, 1.0)
        if show_cworld:
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
        if show_cworld and show_caps_motion:
            # C_world ラベル
            try:
                offs = axis_len_world * 0.12
                cworld_label = getattr(widget, 'cworld_display_name', None) or 'C_world'
                plotter.add_point_labels(
                    np.array([[offs, offs, offs]], dtype=float),
                    [cworld_label],
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
        axis_len_local_base = max(diag * 0.12, 1.0)
        for i, p in enumerate(postures, 1):
            if not caxis_visibility.get(p, True):
                continue
            R_lw = rot['R_local_world'][i - 1]
            O_lw = rot['O_local_world'][i - 1]
            # 各 posture 個別の太さ・長さ係数（FE モードでスライダー操作可能）
            pw = posture_widgets.get(p)
            p_len = float(getattr(pw, 'axis_length_factor', 1.0) or 1.0) if pw else 1.0
            p_rad = float(getattr(pw, 'axis_radius_factor', 1.0) or 1.0) if pw else 1.0
            axis_len_local = axis_len_local_base * p_len
            for k, (col_idx, color) in enumerate((
                (0, '#ff8080'),  # X 軸
                (1, '#8080ff'),  # Y 軸
                (2, '#80ff80'),  # Z 軸
            )):
                try:
                    arrow = pv.Arrow(
                        start=O_lw, direction=R_lw[:, col_idx], scale=axis_len_local,
                        shaft_radius=0.010 * p_rad, tip_radius=0.035 * p_rad, tip_length=0.18,
                    )
                    plotter.add_mesh(arrow, name=f'cloc_{i}_{k}', color=color, opacity=0.85,
                                     pickable=False, reset_camera=False, render=False)
                except Exception:
                    pass
            if not show_caps_motion:
                continue
            try:
                offs = axis_len_local * 0.15
                label_pos = O_lw + np.array([offs, offs, offs])
                # 個別 posture の c_axis_name（FE では QLineEdit で書き換え可能）を使う
                # フォールバックは実際の姿勢番号で（例: posi3 → ラベル "C_x-axis_posi3"）
                try:
                    actual_num = int(p.replace('姿勢', ''))
                except Exception:
                    actual_num = i
                label_text = getattr(pw, 'c_axis_name', None) or f'{local_label_prefix}_posi{actual_num}'
                plotter.add_point_labels(
                    np.array([label_pos], dtype=float),
                    [label_text],
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
        if show_line and show_caps_motion:
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

        # 共有カメラがあればそれを適用、なければデフォルトの bounds fit
        if not self._set_camera_from_shared(widget):
            try:
                plotter.reset_camera(bounds=bounds_all)
            except Exception:
                pass
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

    def _create_posture_view_widget(self, posture_label: str, example_text: str, posture_key: str, axis_letter: str = 'u', is_base_pose: bool = False) -> QWidget:
        widget = DnDWidget()
        widget.posture_key = posture_key
        widget.axis_letter = axis_letter
        widget.is_base_pose = is_base_pose
        widget.posture_tab_label = posture_label  # サブタブの基本名（例 '姿勢1'）
        # 新方式（base へのフィッティングで重ね合わせ）を使うか
        use_base_fit = self._uses_base_fitting(axis_letter)
        widget.use_base_fit = use_base_fit
        # 例: 'C_u-axis_posi1' / base 姿勢は 'C_u-axis_base'
        widget.c_axis_name = f'C_{axis_letter}-axis_{posture_key.replace("posture", "posi")}'
        widget.c_axis_label_prefix = f'C_{axis_letter}-axis'  # 例 'C_u-axis'
        main_layout = QVBoxLayout(widget)

        # Ver.1 風レイアウト: 上段 = 左コントロール + 右 3D ビュー、下段 = ログ
        top_layout = QHBoxLayout()

        # === 左パネル: タイトル / 読み込み / 軸 / 平面サブタブ ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        if is_base_pose:
            title = QLabel('base 姿勢（ALL VIEW の base STL を使用）')
        else:
            title = QLabel(f'{posture_label}  {example_text}')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        left_layout.addWidget(title)

        # base 姿勢は ALL VIEW の base STL を共用するため、STL 読込/領域/フィッティング UI は持たない
        load_btn = QPushButton('STLを読み込む')
        clear_stl_btn = QPushButton('読み込んだSTLを消去')
        clear_stl_btn.setEnabled(False)
        if not is_base_pose:
            left_layout.addWidget(load_btn)
            left_layout.addWidget(clear_stl_btn)
        else:
            load_btn.hide()
            clear_stl_btn.hide()
            note = QLabel('※ base STL は ALL VIEW タブで読み込みます。\nここではその軸用の C_*-axis_base を平面1/2/3で作成します。')
            note.setWordWrap(True)
            note.setStyleSheet('color: #9fb; font-size: 10px;')
            left_layout.addWidget(note)

        # === 2フィット方式の領域 STL + フィッティング UI ===
        # slot 'world'（固定部=C_world）と slot 'link'（動くリンク=C_*-axis）
        widget.region_mesh = None
        widget.region_stl_path = None
        widget.fit_transform = np.eye(4) if is_base_pose else None
        widget.fit_result = None
        widget.region_link_mesh = None
        widget.region_link_stl_path = None
        widget.fit_link_transform = np.eye(4) if is_base_pose else None
        widget.fit_link_result = None

        # 両スロットのボタンを生成（表示は種類により出し分け）
        load_region_btn = QPushButton('固定部(C_world)領域STLを読み込む')
        clear_region_btn = QPushButton('固定部領域STLを消去'); clear_region_btn.setEnabled(False)
        fit_btn = QPushButton('固定部(C_world)へフィッティング'); fit_btn.setEnabled(False)
        fit_check_btn = QPushButton('固定部フィット結果の確認'); fit_check_btn.setEnabled(False)
        load_link_btn = QPushButton('アーム(C_axis)領域STLを読み込む')
        clear_link_btn = QPushButton('アーム領域STLを消去'); clear_link_btn.setEnabled(False)
        fit_link_btn = QPushButton('アーム(C_axis)へフィッティング'); fit_link_btn.setEnabled(False)
        fit_link_check_btn = QPushButton('アームフィット結果の確認'); fit_link_check_btn.setEnabled(False)
        # アームフィットで確定した C_*-axis（このスロットの fit）を消去するボタン
        clear_caxis_btn = QPushButton('C_axis を消去'); clear_caxis_btn.setEnabled(False)
        widget.load_region_btn = load_region_btn
        widget.clear_region_btn = clear_region_btn
        widget.fit_btn = fit_btn
        widget.fit_check_btn = fit_check_btn
        widget.load_link_btn = load_link_btn
        widget.clear_link_btn = clear_link_btn
        widget.fit_link_btn = fit_link_btn
        widget.fit_link_check_btn = fit_link_check_btn
        clear_caxis_btn.setText(f'{widget.c_axis_label_prefix} を消去')
        widget.clear_caxis_btn = clear_caxis_btn

        def _make_param_row(layout, label_text, lo, hi, default, decimals=0, suffix=''):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(110)
            if decimals > 0:
                from PyQt6.QtWidgets import QDoubleSpinBox
                spin = QDoubleSpinBox()
                spin.setDecimals(decimals)
            else:
                spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(default)
            if suffix:
                spin.setSuffix(suffix)
            row.addWidget(lbl)
            row.addWidget(spin, 1)
            layout.addLayout(row)
            return spin

        if use_base_fit and not is_base_pose:
            # 共通フィットパラメータ
            param_group = QGroupBox('フィットパラメータ (RANSAC → ICP)')
            param_layout = QVBoxLayout(param_group)
            voxel_spin = _make_param_row(param_layout, 'voxel サイズ:', 0.0, 1000.0, 0.0, decimals=3, suffix=' mm')
            voxel_spin.setToolTip('0 にすると base の対角長から自動推定します。')
            ransac_iter_spin = _make_param_row(param_layout, 'RANSAC 反復:', 1000, 10000000, 100000)
            icp_iter_spin = _make_param_row(param_layout, 'ICP 反復:', 1, 2000, 50)
            dist_spin = _make_param_row(param_layout, '距離係数(×voxel):', 0.1, 50.0, 1.5, decimals=2)
            widget.fit_voxel_spin = voxel_spin
            widget.fit_ransac_iter_spin = ransac_iter_spin
            widget.fit_icp_iter_spin = icp_iter_spin
            widget.fit_dist_spin = dist_spin
            left_layout.addWidget(param_group)

            # ① 固定部(C_world)フィット
            g1 = QGroupBox('① 固定部(C_world)フィット')
            g1l = QVBoxLayout(g1)
            g1l.addWidget(load_region_btn)
            g1l.addWidget(clear_region_btn)
            g1l.addWidget(fit_btn)
            g1l.addWidget(fit_check_btn)
            fit_status = QLabel('未フィッティング')
            fit_status.setWordWrap(True); fit_status.setStyleSheet('color: #cfa; font-size: 11px;')
            widget.fit_status_label = fit_status
            g1l.addWidget(fit_status)
            left_layout.addWidget(g1)

            # ② アーム(C_*-axis)フィット
            g2 = QGroupBox(f'② アーム({widget.c_axis_label_prefix})フィット')
            g2l = QVBoxLayout(g2)
            g2l.addWidget(load_link_btn)
            g2l.addWidget(clear_link_btn)
            g2l.addWidget(fit_link_btn)
            g2l.addWidget(fit_link_check_btn)
            g2l.addWidget(clear_caxis_btn)
            fit_link_status = QLabel('未フィッティング')
            fit_link_status.setWordWrap(True); fit_link_status.setStyleSheet('color: #cfa; font-size: 11px;')
            widget.fit_link_status_label = fit_link_status
            g2l.addWidget(fit_link_status)
            left_layout.addWidget(g2)
        elif is_base_pose:
            # base 姿勢: アーム領域STL（フィット#2 の基準）を読み込むだけ
            gl = QGroupBox(f'アーム({widget.c_axis_label_prefix})領域（フィット基準）')
            gll = QVBoxLayout(gl)
            load_link_btn.setText('base のアーム領域STLを読み込む')
            clear_link_btn.setText('base のアーム領域STLを消去')
            gll.addWidget(load_link_btn)
            gll.addWidget(clear_link_btn)
            left_layout.addWidget(gl)

        self._add_lighting_toggle(left_layout, widget)

        is_fe = bool(getattr(self, 'fe_mode', False))
        if is_fe:
            build_axis_label = 'ローカル座標系を生成'
            clear_axis_label = 'ローカル座標系を消去'
        else:
            build_axis_label = f'{widget.c_axis_label_prefix} 座標系を生成'
            clear_axis_label = f'{widget.c_axis_label_prefix} 座標系を消去'

        build_axis_btn = QPushButton(build_axis_label)
        build_axis_btn.setEnabled(False)
        clear_axis_btn = QPushButton(clear_axis_label)
        clear_axis_btn.setEnabled(False)
        # 姿勢1〜5（新方式）は C_*-axis を base 側で定義しフィットで運ぶため、
        # 平面ピックによる C_*-axis 生成 UI は持たない。base 姿勢と旧方式(FE)は持つ。
        if use_base_fit and not is_base_pose:
            build_axis_btn.hide()
            clear_axis_btn.hide()
        else:
            left_layout.addWidget(build_axis_btn)
            left_layout.addWidget(clear_axis_btn)

        # FE 専用: 座標系名（キャプション）の自由入力
        axis_name_edit = None
        origin_name_edit = None
        if is_fe:
            # 座標系名（軸の脇に表示するラベル）
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel('座標系名:'))
            axis_name_edit = QLineEdit(widget.c_axis_name)
            axis_name_edit.setPlaceholderText('例: C_u-axis_posi1 / FE_origin / ...')
            name_row.addWidget(axis_name_edit, 1)
            left_layout.addLayout(name_row)

            # 原点名（原点位置に直接表示するラベル、任意）
            origin_name_row = QHBoxLayout()
            origin_name_row.addWidget(QLabel('原点名:'))
            origin_name_edit = QLineEdit('')
            origin_name_edit.setPlaceholderText('原点に表示する名前（任意）')
            origin_name_row.addWidget(origin_name_edit, 1)
            left_layout.addLayout(origin_name_row)
            widget.origin_name = ''

            def _refresh_label_after_text_change(w):
                try:
                    self._render_posture1_plotter(w, reset_view=False)
                except Exception:
                    pass
                try:
                    if getattr(w, 'axis_letter', None) in self.axis_data:
                        mw = self.axis_data[w.axis_letter].get('motion_widget')
                        if mw is not None:
                            self._render_motion_view(mw)
                except Exception:
                    pass

            def _on_axis_name_changed(text, w=widget):
                w.c_axis_name = text or ''
                _refresh_label_after_text_change(w)
            axis_name_edit.textChanged.connect(_on_axis_name_changed)

            def _on_origin_name_changed(text, w=widget):
                w.origin_name = text or ''
                _refresh_label_after_text_change(w)
            origin_name_edit.textChanged.connect(_on_origin_name_changed)

        # FE 専用: 座標軸の太さ・長さ・点・原点球の大きさスライダー
        axis_length_slider = axis_length_spin = None
        axis_radius_slider = axis_radius_spin = None
        point_size_slider = point_size_spin = None
        origin_size_slider = origin_size_spin = None
        if is_fe:
            arrow_ctrl_group = QGroupBox('見た目の調整（このタブのみ）')
            arrow_layout = QVBoxLayout(arrow_ctrl_group)

            # キャプション（文字ラベル）の表示/非表示
            show_caption_cb = QCheckBox('キャプションを表示')
            show_caption_cb.setChecked(True)
            arrow_layout.addWidget(show_caption_cb)
            widget.show_captions = True

            def _on_show_caption_toggled(checked, w=widget):
                w.show_captions = bool(checked)
                try:
                    self._render_posture1_plotter(w, reset_view=False)
                except Exception:
                    pass
            show_caption_cb.toggled.connect(_on_show_caption_toggled)

            def _make_slider_row(label_text, range_max, default_pct):
                row = QHBoxLayout()
                lbl = QLabel(label_text)
                lbl.setMinimumWidth(48)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(10, range_max)
                slider.setValue(default_pct)
                spin = QSpinBox()
                spin.setRange(10, range_max)
                spin.setValue(default_pct)
                spin.setSuffix('%')
                spin.setMinimumWidth(70)
                slider.valueChanged.connect(spin.setValue)
                spin.valueChanged.connect(slider.setValue)
                row.addWidget(lbl)
                row.addWidget(slider, 1)
                row.addWidget(spin)
                return row, slider, spin

            row, axis_length_slider, axis_length_spin = _make_slider_row('軸長さ:', 300, 100)
            arrow_layout.addLayout(row)

            row, axis_radius_slider, axis_radius_spin = _make_slider_row('軸太さ:', 500, 100)
            arrow_layout.addLayout(row)

            # プロットした点（平面ピックの球）の大きさ
            row, point_size_slider, point_size_spin = _make_slider_row('点:', 500, 100)
            arrow_layout.addLayout(row)

            # 原点球の大きさ（C_axis / C_world それぞれの原点）
            row, origin_size_slider, origin_size_spin = _make_slider_row('原点:', 500, 100)
            arrow_layout.addLayout(row)

            left_layout.addWidget(arrow_ctrl_group)

            widget.axis_length_factor = 1.0
            widget.axis_radius_factor = 1.0
            widget.point_size_factor = 1.0
            widget.origin_size_factor = 1.0

            def _refresh_after_visual_change(w):
                try:
                    self._render_posture1_plotter(w, reset_view=False)
                except Exception:
                    pass
                try:
                    if getattr(w, 'axis_letter', None) in self.axis_data:
                        mw = self.axis_data[w.axis_letter].get('motion_widget')
                        if mw is not None:
                            self._render_motion_view(mw)
                except Exception:
                    pass

            def _on_axis_length_changed(v, w=widget):
                w.axis_length_factor = v / 100.0
                _refresh_after_visual_change(w)

            def _on_axis_radius_changed(v, w=widget):
                w.axis_radius_factor = v / 100.0
                _refresh_after_visual_change(w)

            def _on_point_size_changed(v, w=widget):
                w.point_size_factor = v / 100.0
                _refresh_after_visual_change(w)

            def _on_origin_size_changed(v, w=widget):
                w.origin_size_factor = v / 100.0
                _refresh_after_visual_change(w)

            axis_length_slider.valueChanged.connect(_on_axis_length_changed)
            axis_radius_slider.valueChanged.connect(_on_axis_radius_changed)
            point_size_slider.valueChanged.connect(_on_point_size_changed)
            origin_size_slider.valueChanged.connect(_on_origin_size_changed)

        # Ver.2 互換: FE モードでないときも属性は持たせる（描画側のフォールバック用）
        widget.axis_length_factor = getattr(widget, 'axis_length_factor', 1.0)
        widget.axis_radius_factor = getattr(widget, 'axis_radius_factor', 1.0)
        widget.point_size_factor = getattr(widget, 'point_size_factor', 1.0)
        widget.origin_size_factor = getattr(widget, 'origin_size_factor', 1.0)
        widget.origin_name = getattr(widget, 'origin_name', '')
        widget.show_captions = getattr(widget, 'show_captions', True)
        if axis_name_edit is not None:
            widget.axis_name_edit = axis_name_edit
        if origin_name_edit is not None:
            widget.origin_name_edit = origin_name_edit
        if axis_length_slider is not None:
            widget.axis_length_slider = axis_length_slider
            widget.axis_length_spin = axis_length_spin
            widget.axis_radius_slider = axis_radius_slider
            widget.axis_radius_spin = axis_radius_spin
            widget.point_size_slider = point_size_slider
            widget.point_size_spin = point_size_spin
            widget.origin_size_slider = origin_size_slider
            widget.origin_size_spin = origin_size_spin

        # C_world 生成/消去（旧方式: V〜Z軸 / FE のみ表示。新方式では base 側で1回だけ作る）
        build_world_btn = QPushButton('C_world 座標系を生成')
        build_world_btn.setEnabled(False)
        clear_world_btn = QPushButton('C_world 座標系を消去')
        clear_world_btn.setEnabled(False)
        if use_base_fit:
            build_world_btn.hide()
            clear_world_btn.hide()
        else:
            left_layout.addWidget(build_world_btn)
            left_layout.addWidget(clear_world_btn)

        # === 視点記録ボタン ===
        view_btn_row = QHBoxLayout()
        record_view_btn = QPushButton('視点を記録')
        restore_view_btn = QPushButton('記録視点に戻す')
        view_btn_row.addWidget(record_view_btn)
        view_btn_row.addWidget(restore_view_btn)
        left_layout.addLayout(view_btn_row)
        record_view_btn.clicked.connect(lambda: self._record_view(widget))
        restore_view_btn.clicked.connect(lambda: self._restore_recorded_view(widget))

        # === 右パネル: 共有 3D ビュー ===
        if HAS_PYVISTA:
            plotter = QtInteractor(widget)
            self._setup_plotter_jp_fonts(plotter)
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
        widget.display_in_world_frame = False  # posture STL は自前の座標系で表示
        widget.view_kind = 'posture'
        self._attach_camera_observer(widget)

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
        # 姿勢1〜5（新方式）は平面ピックを持たない（C_*-axis は base 定義＋フィットで運ぶ）。
        if use_base_fit and not is_base_pose:
            plane_specs = []
        else:
            plane_specs = [
                # (plane_label, plane_title, system_type)
                ('平面1（XY平面）', f'XY平面 [{cax} 用]', 'c_axis'),
                ('平面2（YZ平面）', f'YZ平面 [{cax} 用]', 'c_axis'),
                ('平面3（ZX平面）', f'ZX平面 [{cax} 用]', 'c_axis'),
            ]
            # 旧方式（FE）のみ C_world 用の W平面サブタブを持つ。
            if not use_base_fit:
                plane_specs += [
                    ('W平面1（XY平面）', 'W平面1（XY平面） [C_world 用]', 'c_world'),
                    ('W平面2（YZ平面）', 'W平面2（YZ平面） [C_world 用]', 'c_world'),
                    ('W平面3（ZX平面）', 'W平面3（ZX平面） [C_world 用]', 'c_world'),
                ]

        plane_widgets = []
        widget.shared_points = {}        # C_*-axis 用
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

        if plane_specs:
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
        widget.clear_stl_btn = clear_stl_btn
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
            widget.clear_stl_btn.setEnabled(True)
            self._save_posture_cache(widget)
            self._update_posture_tab_title(widget)
            if getattr(widget, 'use_base_fit', False):
                self._update_fit_button_state(widget)
            # 新規 STL ロード（非 preserve）はソース変更 → motion-axis を無効化
            if not preserve:
                self._invalidate_motion_for_posture(widget)
            # 同軸の motion-axis を再描画（3 姿勢の STL が揃っていれば描く）
            self._maybe_render_motion_after_mesh_load(widget)

        def _on_load_error(msg: str):
            widget.log_view.append(msg)
            widget.load_btn.setEnabled(True)
            widget.clear_stl_btn.setEnabled(widget.current_mesh is not None)

        load_btn.clicked.connect(lambda: self._open_posture_file(widget))
        clear_stl_btn.clicked.connect(lambda: self._clear_posture_stl(widget))
        build_axis_btn.clicked.connect(lambda: self._build_c_axis(widget))
        clear_axis_btn.clicked.connect(lambda: self._clear_c_axis(widget))
        build_world_btn.clicked.connect(lambda: self._build_c_world_axis(widget))
        clear_world_btn.clicked.connect(lambda: self._clear_c_world_axis(widget))

        if use_base_fit and not is_base_pose:
            # ① 固定部(C_world) フィット
            load_region_btn.clicked.connect(lambda: self._open_region_file(widget, slot='world'))
            clear_region_btn.clicked.connect(lambda: self._clear_region_stl(widget, slot='world'))
            fit_btn.clicked.connect(lambda: self._run_fit(widget, slot='world'))
            fit_check_btn.clicked.connect(lambda: self._show_fit_result(widget, slot='world'))
            # ② アーム(C_*-axis) フィット
            load_link_btn.clicked.connect(lambda: self._open_region_file(widget, slot='link'))
            clear_link_btn.clicked.connect(lambda: self._clear_region_stl(widget, slot='link'))
            fit_link_btn.clicked.connect(lambda: self._run_fit(widget, slot='link'))
            fit_link_check_btn.clicked.connect(lambda: self._show_fit_result(widget, slot='link'))
            clear_caxis_btn.clicked.connect(lambda: self._clear_caxis_fit(widget))
        elif is_base_pose:
            # base 姿勢: リンク領域STL（フィット#2 の基準）の読込/消去のみ
            load_link_btn.clicked.connect(lambda: self._open_region_file(widget, slot='link'))
            clear_link_btn.clicked.connect(lambda: self._clear_region_stl(widget, slot='link'))

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
                        right_clicking=True,
                        show_point=False,
                        pickable_window=False,
                    )
                except Exception:
                    pass
            # アクティブ平面の選択ハイライトを反映するため再描画（カメラは維持）
            self._render_posture1_plotter(widget, reset_view=False)

        widget._on_plane_subtab_changed = _on_plane_subtab_changed
        plane_subtabs.currentChanged.connect(_on_plane_subtab_changed)

        # キャッシュ（点群・C_axis・STLパス）を復元。STL があれば自動読込。
        widget.stl_path = None
        cached_stl_path = self._load_posture_cache(widget)
        for plane_widget in plane_widgets:
            plane_widget._refresh_point_list()
        def _restore_region_slot(slot):
            A = type(self)._FIT_SLOTS[slot]
            p = getattr(widget, A['path'], None)
            if p and os.path.exists(p):
                widget.log_view.append(f'{A["jp"]} キャッシュ検出: {p}')
                self._load_region_stl(widget, p, slot=slot, from_cache=True)
            elif p:
                widget.log_view.append(f'前回の{A["jp"]}STLが見つかりません: {p}')
                setattr(widget, A['path'], None)

        if is_base_pose:
            # base 姿勢は ALL VIEW の base STL を共用。fit は恒等で固定。
            widget.fit_transform = np.eye(4)
            widget.fit_link_transform = np.eye(4)
            base = getattr(self, 'all_view_widget', None)
            base_mesh = getattr(base, 'current_mesh', None) if base else None
            if base_mesh is not None:
                widget.current_mesh = base_mesh
                for pw in widget.plane_widgets:
                    pw.current_mesh = base_mesh
                widget.build_axis_btn.setEnabled(True)
                self._render_posture1_plotter(widget, reset_view=True)
            else:
                self._reset_plotter_placeholder(widget.plotter, 'ALL VIEW で base STL を読み込んでください')
            # base のリンク領域（フィット#2 の基準）を復元
            _restore_region_slot('link')
        else:
            if cached_stl_path and os.path.exists(cached_stl_path):
                widget.log_view.append(f'キャッシュ検出: {cached_stl_path}')
                widget._start_load(cached_stl_path, preserve_state=True)
            elif cached_stl_path:
                widget.log_view.append(f'前回のSTLが見つかりません: {cached_stl_path}')

            # 2スロットの領域 STL キャッシュを復元（新方式のみ）
            if use_base_fit:
                _restore_region_slot('world')
                _restore_region_slot('link')
                self._update_fit_status_label(widget)
                self._update_fit_button_state(widget)

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
                            right_clicking=True,
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

        try:
            self._install_right_click_pick_handler(widget.posture_widget)
        except Exception:
            pass

    def _update_plane_point_buttons(self, widget):
        points = getattr(widget, 'points', [])
        has_points = len(points) > 0
        has_selected = 0 <= getattr(widget, 'selected_point_index', -1) < len(points)
        widget.delete_selected_btn.setEnabled(has_selected)
        widget.delete_last_btn.setEnabled(has_points)
        widget.clear_points_btn.setEnabled(has_points)

    def _get_active_plane_widget(self, posture_widget):
        plane_widgets = getattr(posture_widget, 'plane_widgets', [])
        if not plane_widgets:
            return None
        active_index = getattr(posture_widget, 'active_plane_index', 0)
        if 0 <= active_index < len(plane_widgets):
            return plane_widgets[active_index]
        return None

    def _pick_point_from_plotter(self, plotter, x=None, y=None):
        if plotter is None:
            return None
        iren = getattr(plotter, 'iren', None) or getattr(plotter, 'interactor', None)
        if iren is None:
            return None
        if x is None or y is None:
            try:
                x, y = iren.GetEventPosition()
            except Exception:
                return None

        qt_interactor = getattr(plotter, 'interactor', None)
        ratio = 1.0
        if qt_interactor is not None and hasattr(qt_interactor, 'devicePixelRatioF'):
            try:
                ratio = float(qt_interactor.devicePixelRatioF()) or 1.0
            except Exception:
                ratio = 1.0

        try:
            x = int(round(float(x) * ratio))
            y = int(round(float(y) * ratio))
        except Exception:
            return None

        try:
            renderer = plotter.renderer
        except Exception:
            renderer = None
        if renderer is None:
            return None

        render_window = getattr(plotter, 'ren_win', None)
        if render_window is None:
            render_window = getattr(plotter, 'render_window', None)
        if render_window is not None:
            try:
                _w, h = render_window.GetSize()
                y = max(0, int(h) - 1 - int(y))
            except Exception:
                pass

        picker = getattr(plotter, 'picker', None)
        if picker is None:
            try:
                from vtkmodules.vtkRenderingCore import vtkCellPicker

                picker = vtkCellPicker()
                picker.SetTolerance(0.0005)
            except Exception:
                return None

        try:
            picked = picker.Pick(x, y, 0, renderer)
        except Exception:
            return None
        if not picked:
            return None
        try:
            pos = picker.GetPickPosition()
        except Exception:
            return None
        if pos is None:
            return None
        return np.array(pos, dtype=float)

    def _install_right_click_pick_handler(self, posture_widget):
        plotter = getattr(posture_widget, 'plotter', None)
        if plotter is None or not HAS_PYVISTA:
            return
        if getattr(posture_widget, '_right_click_observer_id', None) is not None:
            return

        qt_interactor = getattr(plotter, 'interactor', None)
        if qt_interactor is None:
            return

        try:
            qt_interactor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        except Exception:
            pass

        class _RightClickFilter(QObject):
            def __init__(self, parent, host):
                super().__init__(parent)
                self._host = host

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
                    w = self._host
                    if getattr(w, 'current_mesh', None) is None:
                        return False
                    active_plane = self._host._get_active_plane_widget(w)
                    if active_plane is None:
                        return False
                    try:
                        pos = event.position()
                        x = int(pos.x())
                        y = int(pos.y())
                    except Exception:
                        try:
                            x = int(event.x())
                            y = int(event.y())
                        except Exception:
                            return False
                    point = self._host._pick_point_from_plotter(w.plotter, x=x, y=y)
                    if point is None:
                        return False
                    self._host._on_plane_surface_point_picked(active_plane, point, force=True)
                elif event.type() == QEvent.Type.KeyPress:
                    try:
                        key = event.key()
                    except Exception:
                        key = None
                    if key == Qt.Key.Key_T:
                        w = self._host
                        self._host._toggle_point_add_mode(w)
                        return True
                return False

        filter_obj = _RightClickFilter(qt_interactor, self)
        qt_interactor.installEventFilter(filter_obj)
        posture_widget._right_click_filter = filter_obj
        posture_widget._right_click_observer_id = 'qt_event_filter'

    def _pick_point_with_normal(self, plotter, mesh, x=None, y=None):
        point = self._pick_point_from_plotter(plotter, x=x, y=y)
        if point is None or mesh is None:
            return None, None
        normal = None
        try:
            normals = getattr(mesh, 'point_normals', None)
            if normals is not None and len(normals):
                idx = int(mesh.find_closest_point(point))
                if 0 <= idx < len(normals):
                    normal = np.array(normals[idx], dtype=float)
        except Exception:
            normal = None
        if normal is None or np.linalg.norm(normal) < 1e-9:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
        return np.array(point, dtype=float), normal

    def _toggle_point_add_mode(self, posture_widget):
        active_plane = self._get_active_plane_widget(posture_widget)
        if active_plane is None:
            return
        btn = getattr(active_plane, 'point_add_btn', None)
        if btn is None:
            return
        try:
            btn.setChecked(not btn.isChecked())
        except Exception:
            pass

    def _on_plane_surface_point_picked(self, widget, point, *_args, force: bool = False):
        if not force and not getattr(widget, 'point_add_enabled', False):
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
        b = posture_widget.current_mesh.bounds
        diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])) or 100.0
        self._configure_lights_for_widget(posture_widget, plotter=plotter)
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

        # FE タブのみ点サイズスライダーが反映される（Ver.2 は factor=1.0）
        pt_factor = float(getattr(posture_widget, 'point_size_factor', 1.0) or 1.0)

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
                    point_size=max(1, int(round(12 * pt_factor))),
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
                    point_size=max(1, int(round(18 * pt_factor))),
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
            bounds = posture_widget.current_mesh.bounds
            if not self._set_camera_from_shared(posture_widget):
                plotter.reset_camera(bounds=bounds)
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
                    right_clicking=True,
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
            top = settings.setdefault(self.settings_top_key, {})
            axis_section = top.setdefault(f'{axis_letter}_axis', {})
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

            # 任意領域 STL パス
            region_path = getattr(posture_widget, 'region_stl_path', None)
            if region_path:
                posture_entry['region_stl_path'] = region_path
            else:
                posture_entry.pop('region_stl_path', None)

            # 両スロットのフィッティング変換（4x4）と統計、リンク領域パスを保存
            for slot, A in type(self)._FIT_SLOTS.items():
                # リンクスロットの領域パス（world は既に region_stl_path で保存済み）
                if slot == 'link':
                    lp = getattr(posture_widget, A['path'], None)
                    if lp:
                        posture_entry['region_link_stl_path'] = lp
                    else:
                        posture_entry.pop('region_link_stl_path', None)
                fit_T = getattr(posture_widget, A['T'], None)
                tkey = 'fit_transform' if slot == 'world' else 'fit_link_transform'
                rkey = 'fit_result' if slot == 'world' else 'fit_link_result'
                if fit_T is not None:
                    posture_entry[tkey] = np.asarray(fit_T, dtype=float).tolist()
                    fr = getattr(posture_widget, A['res'], None)
                    if isinstance(fr, dict):
                        posture_entry[rkey] = {
                            k: fr.get(k) for k in (
                                'ransac_fitness', 'ransac_rmse',
                                'icp_fitness', 'icp_rmse', 'voxel_size',
                            )
                        }
                else:
                    posture_entry.pop(tkey, None)
                    posture_entry.pop(rkey, None)

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
                ((settings.get(self.settings_top_key) or {}).get(f'{axis_letter}_axis') or {}).get(posture_key) or {}
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

        # 両スロットの領域 STL パスとフィッティング変換を復元
        posture_widget.region_stl_path = posture_entry.get('region_stl_path') or None
        posture_widget.region_link_stl_path = posture_entry.get('region_link_stl_path') or None
        for slot, A in type(self)._FIT_SLOTS.items():
            tkey = 'fit_transform' if slot == 'world' else 'fit_link_transform'
            rkey = 'fit_result' if slot == 'world' else 'fit_link_result'
            fit_T = posture_entry.get(tkey)
            if fit_T is not None:
                try:
                    setattr(posture_widget, A['T'], np.asarray(fit_T, dtype=float))
                    setattr(posture_widget, A['res'], posture_entry.get(rkey) or {})
                except Exception:
                    setattr(posture_widget, A['T'], None)
                    setattr(posture_widget, A['res'], None)
            else:
                setattr(posture_widget, A['T'], None)
                setattr(posture_widget, A['res'], None)

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
        self._invalidate_motion_for_posture(posture_widget)

    def _build_c_world_axis(self, posture_widget):
        plane_keys = ['W平面1（XY平面）', 'W平面2（YZ平面）', 'W平面3（ZX平面）']
        pts = self._collect_plane_points(posture_widget.shared_world_points, plane_keys)
        result = self._compute_axis_system(pts, posture_widget.log_view, prefix='C_world')
        if result is None:
            return
        # 呂（FE）タブ専用: C_world の Z 軸（緑）の向きを反転する仕様
        if getattr(self, 'fe_mode', False):
            try:
                result['ez'] = -np.asarray(result['ez'], dtype=float)
                result['raw_z'] = -np.asarray(result['raw_z'], dtype=float)
                posture_widget.log_view.append(
                    '【FE 仕様】C_world の Z 軸（緑）の向きを反転しました。'
                )
            except Exception:
                pass
        posture_widget.c_world = result
        posture_widget.clear_world_btn.setEnabled(True)
        self._render_posture1_plotter(posture_widget, reset_view=False)
        # C_world ができたので共有カメラを反映
        self._apply_shared_camera_with_render(posture_widget)
        self._save_posture_cache(posture_widget)
        self._invalidate_motion_for_posture(posture_widget)

    def _clear_c_axis(self, posture_widget):
        if getattr(posture_widget, 'c_axis', None) is None:
            return
        posture_widget.c_axis = None
        posture_widget.clear_axis_btn.setEnabled(False)
        prefix = getattr(posture_widget, 'c_axis_label_prefix', 'C_u-axis')
        posture_widget.log_view.append(f'{prefix} 座標系を消去しました。')
        self._render_posture1_plotter(posture_widget, reset_view=False)
        self._save_posture_cache(posture_widget)
        self._invalidate_motion_for_posture(posture_widget)

    def _clear_c_world_axis(self, posture_widget):
        if getattr(posture_widget, 'c_world', None) is None:
            return
        posture_widget.c_world = None
        posture_widget.clear_world_btn.setEnabled(False)
        posture_widget.log_view.append('C_world 座標系を消去しました。')
        self._render_posture1_plotter(posture_widget, reset_view=False)
        self._save_posture_cache(posture_widget)
        self._invalidate_motion_for_posture(posture_widget)

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

        # c_axis（ローカル座標系）にのみユーザー定義の太さ・長さ係数を適用。
        # c_world は常に既定値（FE タブでも変更されない）。
        len_factor = float(getattr(posture_widget, 'axis_length_factor', 1.0) or 1.0)
        rad_factor = float(getattr(posture_widget, 'axis_radius_factor', 1.0) or 1.0)
        show_caps = bool(getattr(posture_widget, 'show_captions', True))

        systems = [
            {
                'key': 'c_axis',
                'frame': getattr(posture_widget, 'c_axis', None),
                'label': getattr(posture_widget, 'c_axis_name', 'C_u-axis'),
                'label_color': '#ffffaa',
                'origin_color': '#ffff66',
                'arrow_colors': ('#ff3030', '#3060ff', '#30c030'),  # X red / Y blue / Z green
                'actor_prefix': 'c_axis',
                'length_factor': len_factor,
                'radius_factor': rad_factor,
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
                'length_factor': 1.0,
                'radius_factor': 1.0,
            },
        ]

        for sys_def in systems:
            prefix = sys_def['actor_prefix']
            # 既存アクター除去
            for suffix in ('_x', '_y', '_z', '_origin', '_label', '_origin_label'):
                try:
                    plotter.remove_actor(prefix + suffix)
                except Exception:
                    pass

            frame = sys_def['frame']
            if frame is None:
                continue

            origin = frame['origin']
            cx, cy, cz = sys_def['arrow_colors']
            this_axis_len = axis_len * sys_def.get('length_factor', 1.0)
            r_f = sys_def.get('radius_factor', 1.0)
            for suffix, vec, color in (
                ('_x', frame['ex'], cx),
                ('_y', frame['ey'], cy),
                ('_z', frame['ez'], cz),
            ):
                try:
                    arrow = pv.Arrow(
                        start=origin,
                        direction=vec,
                        scale=this_axis_len,
                        shaft_radius=0.015 * r_f,
                        tip_radius=0.045 * r_f,
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
                origin_factor = float(getattr(posture_widget, 'origin_size_factor', 1.0) or 1.0)
                plotter.add_mesh(
                    pv.PolyData(np.array([origin], dtype=float)),
                    name=prefix + '_origin',
                    color=sys_def['origin_color'],
                    point_size=max(1, int(round(14 * origin_factor))),
                    render_points_as_spheres=True,
                    style='points',
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass

            # 座標系名のラベル（原点から軸長 10% オフセット）
            label_offset = this_axis_len * 0.10
            label_pos = np.array(origin, dtype=float) + np.array(
                [label_offset, label_offset, label_offset], dtype=float
            )
            if show_caps:
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

            # 原点名のラベル（FE タブ専用、c_axis にのみ表示。原点位置に直接配置）
            if show_caps and prefix == 'c_axis':
                origin_name = (getattr(posture_widget, 'origin_name', '') or '').strip()
                if origin_name:
                    # 軸ラベルと重ならないよう、反対方向に少しオフセット
                    op_offset = this_axis_len * 0.06
                    op_pos = np.array(origin, dtype=float) - np.array(
                        [op_offset, op_offset, op_offset], dtype=float
                    )
                    try:
                        plotter.add_point_labels(
                            np.array([op_pos], dtype=float),
                            [origin_name],
                            name=prefix + '_origin_label',
                            font_size=16,
                            text_color='#ffffff',
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

    # ===== 任意領域 STL + base へのフィッティング（2フィット方式）=====
    # slot='world' … 固定部(C_world)領域、slot='link' … 動くアーム(C_*-axis)領域
    _FIT_SLOTS = {
        'world': {
            'mesh': 'region_mesh', 'path': 'region_stl_path',
            'T': 'fit_transform', 'res': 'fit_result',
            'clear_btn': 'clear_region_btn', 'fit_btn': 'fit_btn',
            'check_btn': 'fit_check_btn', 'status': 'fit_status_label',
            'jp': '固定部(C_world)領域',
        },
        'link': {
            'mesh': 'region_link_mesh', 'path': 'region_link_stl_path',
            'T': 'fit_link_transform', 'res': 'fit_link_result',
            'clear_btn': 'clear_link_btn', 'fit_btn': 'fit_link_btn',
            'check_btn': 'fit_link_check_btn', 'status': 'fit_link_status_label',
            'jp': 'アーム(C_axis)領域',
        },
    }

    def _get_base_region_mesh(self):
        """固定部(world)フィットの基準: ALL VIEW(base) の固定部領域メッシュ。"""
        base = getattr(self, 'all_view_widget', None)
        if base is None:
            return None
        return getattr(base, 'region_mesh', None)

    def _get_base_link_region_mesh(self, axis_letter):
        """アーム(link)フィットの基準: 各軸 base 姿勢タブのアーム領域メッシュ。"""
        ad = self.axis_data.get(axis_letter)
        bw = ad.get('base_widget') if ad else None
        return getattr(bw, 'region_link_mesh', None) if bw else None

    def _fit_target_mesh(self, widget, slot):
        if slot == 'link':
            return self._get_base_link_region_mesh(getattr(widget, 'axis_letter', None))
        return self._get_base_region_mesh()

    def _propagate_base_mesh_to_poses(self):
        """ALL VIEW の base STL を、各軸の base 姿勢タブへ反映して再描画する。"""
        base = getattr(self, 'all_view_widget', None)
        mesh = getattr(base, 'current_mesh', None) if base else None
        for ad in self.axis_data.values():
            bw = ad.get('base_widget')
            if bw is None:
                continue
            bw.current_mesh = mesh
            for pw in getattr(bw, 'plane_widgets', []):
                pw.current_mesh = mesh
            if mesh is not None:
                bw.fit_transform = np.eye(4)
                bw.build_axis_btn.setEnabled(True)
                self._render_posture1_plotter(bw, reset_view=True)
            else:
                bw.build_axis_btn.setEnabled(False)
                self._reset_plotter_placeholder(bw.plotter, 'ALL VIEW で base STL を読み込んでください')
            # base 姿勢の mesh 変化は motion を無効化
            self._invalidate_motion_for_posture(bw)

    def _refresh_all_fit_buttons(self):
        """全姿勢のフィッティングボタン状態を更新（base 領域の有無が変わったとき用）。"""
        for ad in self.axis_data.values():
            for pw in (ad.get('posture_widgets') or {}).values():
                if getattr(pw, 'use_base_fit', False):
                    self._update_fit_button_state(pw)

    def _update_fit_button_state(self, widget):
        if not getattr(widget, 'use_base_fit', False):
            return
        for slot, A in type(self)._FIT_SLOTS.items():
            has_region = getattr(widget, A['mesh'], None) is not None
            has_target = self._fit_target_mesh(widget, slot) is not None
            fb = getattr(widget, A['fit_btn'], None)
            cb = getattr(widget, A['check_btn'], None)
            clr = getattr(widget, A['clear_btn'], None)
            if fb is not None:
                fb.setEnabled(bool(has_region and has_target and HAS_OPEN3D))
            if cb is not None:
                cb.setEnabled(getattr(widget, A['T'], None) is not None)
            if clr is not None:
                clr.setEnabled(has_region)
        # 「C_*-axis を消去」はアーム(link)フィットが確定している時のみ有効
        ccb = getattr(widget, 'clear_caxis_btn', None)
        if ccb is not None:
            ccb.setEnabled(getattr(widget, 'fit_link_transform', None) is not None)

    def _clear_caxis_fit(self, widget):
        """アームフィットで確定した C_*-axis（link フィット結果）のみを消去する。
        アーム領域STLは保持するので、再度②アームフィットすれば復活する。"""
        widget.fit_link_transform = None
        widget.fit_link_result = None
        prefix = getattr(widget, 'c_axis_label_prefix', 'C_axis')
        widget.log_view.append(
            f'{prefix} を消去しました。再度「②アーム({prefix})へフィッティング」を実行してください。'
        )
        self._save_posture_cache(widget)
        self._update_fit_status_label(widget, 'link')
        self._update_fit_button_state(widget)
        self._invalidate_motion_for_posture(widget)

    def _update_fit_status_label(self, widget, slot=None):
        slots = [slot] if slot else list(type(self)._FIT_SLOTS.keys())
        for s in slots:
            A = type(self)._FIT_SLOTS[s]
            lbl = getattr(widget, A['status'], None)
            if lbl is None:
                continue
            if getattr(widget, A['T'], None) is None:
                lbl.setText('未フィッティング')
                continue
            fr = getattr(widget, A['res'], None) or {}
            lbl.setText(
                'フィット済み: '
                f"ICP fitness={fr.get('icp_fitness', float('nan')):.3f}, "
                f"RMSE={fr.get('icp_rmse', float('nan')):.3f} mm"
            )

    def _read_region_mesh(self, path):
        mesh = pv.read(path)
        mesh = mesh.extract_surface(algorithm='dataset_surface').triangulate()
        mesh = mesh.compute_normals(
            cell_normals=False, point_normals=True,
            consistent_normals=True, auto_orient_normals=True,
        )
        return mesh

    def _open_region_file(self, widget, slot='world'):
        jp = type(self)._FIT_SLOTS[slot]['jp']
        path, _ = QFileDialog.getOpenFileName(widget, f'{jp} STLファイルを開く', '', 'STL Files (*.stl)')
        if not path:
            return
        self._load_region_stl(widget, path, slot=slot)

    def _load_region_stl(self, widget, path, slot='world', from_cache: bool = False):
        if not HAS_PYVISTA:
            widget.log_view.append('pyvista が無いため任意領域を読み込めません。')
            return
        A = type(self)._FIT_SLOTS[slot]
        try:
            mesh = self._read_region_mesh(path)
        except Exception as e:
            widget.log_view.append(f'{A["jp"]} STL読み込み失敗: {e}')
            return
        setattr(widget, A['mesh'], mesh)
        setattr(widget, A['path'], path)
        widget.log_view.append(
            f'{A["jp"]} STLを読み込みました: {path} (points={mesh.n_points})'
        )
        clr = getattr(widget, A['clear_btn'], None)
        if clr is not None:
            clr.setEnabled(True)
        if not from_cache:
            self._save_posture_cache(widget)
        # 基準(target)側の領域が変わると、その軸の全姿勢のフィットボタンに影響
        view_kind = getattr(widget, 'view_kind', None)
        is_base_target = (view_kind == 'all_view') or getattr(widget, 'is_base_pose', False)
        if is_base_target:
            self._refresh_all_fit_buttons()
        else:
            self._update_fit_button_state(widget)

    def _clear_region_stl(self, widget, slot='world'):
        A = type(self)._FIT_SLOTS[slot]
        setattr(widget, A['mesh'], None)
        setattr(widget, A['path'], None)
        setattr(widget, A['T'], None)
        setattr(widget, A['res'], None)
        widget.log_view.append(f'{A["jp"]} STLを消去しました。フィット結果もクリアしました。')
        clr = getattr(widget, A['clear_btn'], None)
        if clr is not None:
            clr.setEnabled(False)
        self._save_posture_cache(widget)
        self._update_fit_status_label(widget, slot)
        view_kind = getattr(widget, 'view_kind', None)
        is_base_target = (view_kind == 'all_view') or getattr(widget, 'is_base_pose', False)
        if is_base_target:
            self._refresh_all_fit_buttons()
        else:
            self._update_fit_button_state(widget)
            self._invalidate_motion_for_posture(widget)

    def _run_fit(self, widget, slot='world'):
        if not HAS_OPEN3D:
            widget.log_view.append('open3d が見つかりません。`pip install open3d` を実行してください。')
            return
        A = type(self)._FIT_SLOTS[slot]
        source = getattr(widget, A['mesh'], None)
        target = self._fit_target_mesh(widget, slot)
        if source is None:
            widget.log_view.append(f'この姿勢の{A["jp"]}STLが未読込です。')
            return
        if target is None:
            if slot == 'link':
                widget.log_view.append('base のアーム領域STLが未読込です。base 姿勢タブで読み込んでください。')
            else:
                widget.log_view.append('base の固定部領域STLが未読込です。ALL VIEW で読み込んでください。')
            return
        params = {
            'voxel_size': float(widget.fit_voxel_spin.value()),
            'ransac_iter': int(widget.fit_ransac_iter_spin.value()),
            'icp_iter': int(widget.fit_icp_iter_spin.value()),
            'dist_factor': float(widget.fit_dist_spin.value()),
        }
        widget.log_view.append(f'=== {A["jp"]} フィッティング開始（RANSAC → ICP） ===')
        fb = getattr(widget, A['fit_btn'], None)
        if fb is not None:
            fb.setEnabled(False)

        thread = QThread(widget)
        worker = FitWorker(source, target, params)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(lambda m: widget.log_view.append(m))
        worker.finished.connect(lambda res, s=slot: self._on_fit_finished(widget, res, s))
        worker.error.connect(lambda m, s=slot: self._on_fit_error(widget, m, s))
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        setattr(widget, f'_fit_thread_{slot}', thread)
        setattr(widget, f'_fit_worker_{slot}', worker)
        thread.start()

    def _on_fit_finished(self, widget, res, slot='world'):
        A = type(self)._FIT_SLOTS[slot]
        T = np.asarray(res['transform'], dtype=float)
        setattr(widget, A['T'], T)
        setattr(widget, A['res'], res)
        widget.log_view.append(f'=== {A["jp"]} フィッティング完了 ===')
        widget.log_view.append(
            f"  RANSAC: fitness={res['ransac_fitness']:.4f}, RMSE={res['ransac_rmse']:.4f} mm"
        )
        widget.log_view.append(
            f"  ICP   : fitness={res['icp_fitness']:.4f}, RMSE={res['icp_rmse']:.4f} mm"
        )
        widget.log_view.append('  変換行列 T (姿勢STL座標 → base座標):')
        for r in range(4):
            widget.log_view.append(
                f'    [{T[r,0]:+.5f}, {T[r,1]:+.5f}, {T[r,2]:+.5f}, {T[r,3]:+.4f}]'
            )
        self._update_fit_status_label(widget, slot)
        self._update_fit_button_state(widget)
        fb = getattr(widget, A['fit_btn'], None)
        if fb is not None:
            fb.setEnabled(True)
        self._save_posture_cache(widget)
        self._invalidate_motion_for_posture(widget)
        # 固定部フィットは STL↔C_world を結ぶので照明・視点を同期反映
        if slot == 'world' and getattr(widget, 'current_mesh', None) is not None:
            try:
                self._render_posture1_plotter(widget, reset_view=False)
            except Exception:
                pass
            if self.shared_camera_world is not None:
                self._apply_shared_camera_with_render(widget)
        self._show_fit_result(widget, slot)

    def _on_fit_error(self, widget, msg, slot='world'):
        widget.log_view.append(msg)
        fb = getattr(widget, type(self)._FIT_SLOTS[slot]['fit_btn'], None)
        if fb is not None:
            fb.setEnabled(True)

    def _show_fit_result(self, widget, slot='world'):
        """基準領域（灰）と、変換後の姿勢領域（橙）を別ウィンドウで重ね表示。"""
        if not HAS_PYVISTA:
            widget.log_view.append('pyvista が無いため結果を表示できません。')
            return
        A = type(self)._FIT_SLOTS[slot]
        T = getattr(widget, A['T'], None)
        source = getattr(widget, A['mesh'], None)
        target = self._fit_target_mesh(widget, slot)
        if T is None or source is None or target is None:
            widget.log_view.append('フィッティング結果を表示できません（結果または領域STLが不足）。')
            return
        from PyQt6.QtWidgets import QDialog
        title = f'{getattr(widget, "posture_tab_label", "姿勢")} / {A["jp"]}'
        dlg = QDialog(widget)
        dlg.setWindowTitle(f'フィッティング結果: {title}')
        dlg.resize(880, 660)
        lay = QVBoxLayout(dlg)
        info = QLabel(f'灰: base の{A["jp"]} / 橙: フィッティング後の姿勢領域')
        lay.addWidget(info)
        plotter = QtInteractor(dlg)
        self._setup_plotter_jp_fonts(plotter)
        background_color, _ = self._load_visual_settings()
        plotter.set_background(background_color, top=self._background_top_color(background_color))
        self._configure_lights(plotter)
        try:
            plotter.add_mesh(target.copy(), color='#b0b0b0', opacity=0.55,
                             smooth_shading=True, name='base_region')
            moved = source.copy()
            moved.transform(np.asarray(T, dtype=float), inplace=True)
            plotter.add_mesh(moved, color='#ff9030', opacity=0.65,
                             smooth_shading=True, name='posture_region')
            plotter.reset_camera()
        except Exception as e:
            widget.log_view.append(f'結果表示中にエラー: {e}')
        lay.addWidget(plotter.interactor)

        def _on_close(_ev, p=plotter):
            try:
                p.close()
            except Exception:
                pass
        dlg.finished.connect(lambda _r: _on_close(None))
        if not hasattr(self, '_fit_dialogs'):
            self._fit_dialogs = []
        self._fit_dialogs.append(dlg)
        dlg.show()


class FEAxisWidget(Tab2Widget):
    """「呂」タブの中身。

    構造:
      呂（MainWindow のトップタブ）
        └ FE軸検証（このウィジェットがホストするサブタブ）
             └ 姿勢1 / 姿勢2 / 姿勢3 / U軸回転軸

    Ver.2 の U axis と同じ内部構造を、完全に独立した状態・キャッシュ
    （settings.json のトップキーは 'tab2_fe'）で提供する。
    """

    SETTINGS_TOP_KEY = 'tab2_fe'

    def __init__(self, parent=None):
        # Tab2Widget.__init__ は ALL VIEW + 全 6 軸を生成するためバイパスし、
        # ここでは「FE軸検証」サブタブ 1 枚だけを構築する。
        QWidget.__init__(self, parent)
        self.settings_top_key = type(self).SETTINGS_TOP_KEY
        # FE 専用 UI を有効化（_create_axis_tab 呼び出しより前にセット）
        self.fe_mode = True
        self.axis_data = {}
        self.visual_widgets = []
        self.lighting_checkboxes = []
        self.shared_camera_world = None
        self.recorded_view_world = None
        self.all_view_widget = None
        # 共有カメラ・記録視点は別のキー名で読み書きされるので、main の Ver.2 とは独立
        self._load_shared_camera_cache()
        self._load_recorded_view_cache()
        register_lighting_listener(self._on_global_lighting_changed)

        layout = QVBoxLayout(self)
        # 「FE軸検証」というラベルのサブタブを 1 つだけ持つ QTabWidget を配置。
        # サブタブの中身は U axis（姿勢1/2/3 + U軸回転軸）と同等。
        self.top_subtabs = QTabWidget()
        axis_widget = self._create_axis_tab('u', joint_type='rotation')
        self.top_subtabs.addTab(axis_widget, 'FE軸検証')
        layout.addWidget(self.top_subtabs)
        self.setLayout(layout)
