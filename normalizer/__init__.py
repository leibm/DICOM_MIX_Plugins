# -*- coding: utf-8 -*-
"""
Normalizer Plugin for DICOM MIX Tools
DICOM 异构断层数据归一化插件

依赖: pydicom, numpy
"""

import os
import logging
from typing import List

logger = logging.getLogger("plugin.normalizer")

PLUGIN_INFO = {
    "name": "normalizer",
    "version": "1.0.0",
    "description": "DICOM 异构断层数据归一化插件",
    "author": "leibm",
    "dependencies": ["pydicom", "numpy"],
    "entry_point": "NormalizerPlugin",
}


class NormalizerPlugin:
    """
    归一化插件主类。

    将来自不同品牌 DSA/CT 设备的三维断层序列归一化为标准 CT Image Storage 格式。
    """

    def __init__(self, app_controller=None, **kwargs):
        self._app = app_controller
        logger.info("Normalizer Plugin initialized")

    def is_available(self) -> bool:
        """检查运行环境是否满足。"""
        try:
            import pydicom
            import numpy
            return True
        except ImportError:
            return False

    def activate(self):
        """激活插件：弹出归一化对话框。"""
        if self._app is None:
            return
        # 调用主程序已有的归一化功能
        window = getattr(self._app, "window", None)
        if window and hasattr(window, "_on_show_normalizer"):
            window._on_show_normalizer()
        else:
            self._show_message("归一化功能不可用")

    def deactivate(self):
        """停用插件。"""
        pass

    def _show_message(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        if self._app and hasattr(self._app, "window"):
            QMessageBox.information(self._app.window, "归一化插件", msg)
        else:
            print(f"[Normalizer] {msg}")
