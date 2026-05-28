import json
import os
from pathlib import Path

_LIGHTING_KEY = 'lighting_enabled'
_lighting_enabled_cache = None
_lighting_listeners = []


def _get_settings_path() -> Path:
    base = os.getenv('APPDATA') or str(Path.home())
    folder = Path(base) / 'FRS-Simulator'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / 'settings.json'


def load_settings() -> dict:
    path = _get_settings_path()
    if not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d: dict):
    path = _get_settings_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def export_points_data(mode_points: dict, filepath: str) -> bool:
    """点データをJSONファイルにエクスポート"""
    try:
        data = {
            'mode_points': {
                mode: [[float(p[0]), float(p[1]), float(p[2])] for p in points]
                for mode, points in mode_points.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def import_points_data(filepath: str) -> dict:
    """JSONファイルから点データをインポート"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('mode_points', {})
    except Exception:
        return {}


def apply_theme(app, dark: bool):
    if not app:
        return
    if dark:
        dark_style = """
        QWidget { background-color: #2e2e2e; color: #eaeaea; }
        QPushButton { background-color: #444444; color: #ffffff; border: 1px solid #666666; padding: 4px; }
        QLineEdit, QTextEdit, QListWidget { background-color: #3a3a3a; color: #ffffff; }
        QTabWidget::pane { background: #2e2e2e; }
        """
        app.setStyleSheet(dark_style)
    else:
        app.setStyleSheet('')


def get_lighting_enabled() -> bool:
    global _lighting_enabled_cache
    if _lighting_enabled_cache is None:
        settings = load_settings() or {}
        _lighting_enabled_cache = bool(settings.get(_LIGHTING_KEY, True))
    return bool(_lighting_enabled_cache)


def set_lighting_enabled(enabled: bool) -> None:
    global _lighting_enabled_cache
    _lighting_enabled_cache = bool(enabled)
    settings = load_settings() or {}
    settings[_LIGHTING_KEY] = bool(enabled)
    save_settings(settings)
    _notify_lighting_listeners(bool(enabled))


def register_lighting_listener(callback) -> None:
    if not callable(callback):
        return
    if callback not in _lighting_listeners:
        _lighting_listeners.append(callback)


def _notify_lighting_listeners(enabled: bool) -> None:
    for cb in list(_lighting_listeners):
        try:
            cb(bool(enabled))
        except Exception:
            pass
