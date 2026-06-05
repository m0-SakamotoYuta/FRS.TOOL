import json
import os
import shutil
from datetime import date
from pathlib import Path

_LIGHTING_KEY = 'lighting_enabled'
_lighting_enabled_cache = None
_lighting_listeners = []

# 日付付きバックアップの保持日数
_DAILY_BACKUP_RETAIN_DAYS = 30


def _get_settings_path() -> Path:
    base = os.getenv('APPDATA') or str(Path.home())
    folder = Path(base) / 'FRS-Simulator'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / 'settings.json'


def _get_daily_backup_dir() -> Path:
    folder = _get_settings_path().parent / 'daily_backups'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def ensure_startup_backup() -> None:
    """起動時に 1 回だけ呼ぶ。現在の settings.json をセッション開始時のスナップショットとして保存する。

    `settings.json.startup` は毎回上書きするが、これは「起動時点の状態」を表すため、
    セッション中にどんな破壊的変更が起きても起動時点に戻せる。
    日付付きバックアップ（daily_backups/settings-YYYY-MM-DD.json）も同時に作成し、
    その日の最初の起動時の状態を 1 日 1 ファイルとして長期保存する。
    """
    path = _get_settings_path()
    if not path.exists():
        return
    # 1) 起動時スナップショット（毎回上書き、固定名）
    try:
        shutil.copy2(str(path), str(path.with_suffix('.json.startup')))
    except Exception:
        pass
    # 2) 日付付きバックアップ（1 日 1 ファイル、その日の最初の起動時だけ作成）
    try:
        daily_dir = _get_daily_backup_dir()
        today_file = daily_dir / f'settings-{date.today().isoformat()}.json'
        if not today_file.exists():
            shutil.copy2(str(path), str(today_file))
        # 古い日付付きバックアップを掃除（保持日数を超えたものを削除）
        _prune_daily_backups(daily_dir)
    except Exception:
        pass


def _prune_daily_backups(daily_dir: Path) -> None:
    try:
        files = sorted(
            (p for p in daily_dir.glob('settings-*.json') if p.is_file()),
            key=lambda p: p.name,
        )
        if len(files) > _DAILY_BACKUP_RETAIN_DAYS:
            for old in files[:-_DAILY_BACKUP_RETAIN_DAYS]:
                try:
                    old.unlink()
                except Exception:
                    pass
    except Exception:
        pass


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
    # 保存前にバックアップを取得（直近5世代まで保持）
    try:
        if path.exists():
            for i in range(4, 0, -1):
                src = path.with_suffix(f'.json.bak{i}')
                dst = path.with_suffix(f'.json.bak{i+1}')
                if src.exists():
                    shutil.move(str(src), str(dst))
            shutil.copy2(str(path), str(path.with_suffix('.json.bak1')))
    except Exception:
        pass
    try:
        # atomic write: 一時ファイルに書いてから rename
        tmp_path = path.with_suffix('.json.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp_path), str(path))
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
