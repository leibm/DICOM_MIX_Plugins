# -*- coding: utf-8 -*-
"""
3D Renderer Plugin for DICOM MIX Tools
三维医学图像体渲染与 MPR 重建插件

依赖: vtk, pyvista, pydicom, numpy
"""

import os
import sys
import logging
from typing import List, Optional

# 显式指定 Qt 后端为 PySide6，避免 pyvistaqt 无法找到绑定
os.environ["QT_API"] = "pyside6"

logger = logging.getLogger("plugin.3d_renderer")

PLUGIN_INFO = {
    "name": "3d_renderer",
    "version": "1.0.0",
    "description": "三维医学图像体渲染与 MPR 重建插件",
    "author": "leibm",
    "dependencies": ["vtk", "pyvista"],
    "entry_point": "RendererPlugin",
}


class RendererPlugin:
    """
    3D 渲染插件主类。

    提供功能:
    - 体渲染 (Volume Rendering)
    - 面渲染 (Marching Cubes / Surface Rendering)
    - MPR 多平面重建 (Multi-Planar Reconstruction)
    """

    def __init__(self, app_controller=None, **kwargs):
        self._app = app_controller
        self._window = None
        self._plotter = None
        logger.info("3D Renderer Plugin initialized")

    def is_available(self) -> bool:
        """检查运行环境是否满足（VTK/PyVista 是否已安装）。"""
        try:
            import vtk
            import pyvista
            return True
        except ImportError:
            return False

    def activate(self):
        """激活插件：显示 3D 渲染窗口。"""
        if not self.is_available():
            self._show_missing_deps_dialog()
            return

        # 获取当前选中的 DICOM 文件列表
        file_list = self._get_selected_files()
        if not file_list:
            self._show_message("请先在左侧数据源列表中选中一个断层序列")
            return

        self._show_render_window(file_list)

    def deactivate(self):
        """停用插件：关闭渲染窗口。"""
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
            self._plotter = None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_selected_files(self) -> List[str]:
        """从主窗口获取当前选中的 DICOM 文件列表。"""
        if self._app is None:
            return []
        window = getattr(self._app, "window", None)
        if window is None:
            return []

        # 获取树形控件当前选中的节点
        from PySide6.QtCore import QModelIndex
        current = window.tree_view.selectionModel().currentIndex()
        if not current.isValid():
            return []

        item = current.internalPointer()
        if not item or not hasattr(item, 'data'):
            return []

        data = item.data
        node_type = data.get("type", "")
        file_list = []

        if "影像" in node_type:
            fpath = data.get("file_path", "")
            if fpath and os.path.isfile(fpath):
                file_list = [fpath]
        elif "序列" in node_type:
            file_list = data.get("instances", [])
        elif "检查" in node_type:
            if hasattr(item, 'children'):
                for child in item.children:
                    child_data = getattr(child, 'data', {})
                    file_list.extend(child_data.get("instances", []))

        return file_list

    def _show_message(self, msg: str):
        """显示消息提示。"""
        if self._app and hasattr(self._app, "window"):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self._app.window, "3D 渲染", msg)
        else:
            print(f"[3D Renderer] {msg}")

    def _show_missing_deps_dialog(self):
        """提示用户安装缺失的依赖。"""
        from PySide6.QtWidgets import QMessageBox
        msg = (
            "3D 渲染插件需要以下依赖:\n"
            "  - vtk\n"
            "  - pyvista\n\n"
            "请运行以下命令安装:\n"
            "  pip install vtk pyvista\n\n"
            "安装完成后重新激活插件。"
        )
        if self._app and hasattr(self._app, "window"):
            QMessageBox.warning(self._app.window, "缺少依赖", msg)
        else:
            print(f"[3D Renderer] {msg}")

    def _show_render_window(self, file_list: List[str]):
        """创建并显示 3D 渲染窗口。"""
        import numpy as np
        import pydicom
        import pyvista as pv
        from pyvistaqt import QtInteractor

        # 读取并排序 DICOM 文件
        try:
            volume_data, spacing = self._load_dicom_volume(file_list)
        except Exception as e:
            self._show_message(f"加载 DICOM 数据失败:\n{e}")
            return

        # 创建 PyVista Qt 交互窗口
        self._plotter = QtInteractor()

        # 添加体渲染
        grid = pv.ImageData()
        grid.dimensions = volume_data.shape
        grid.spacing = spacing
        grid.origin = (0, 0, 0)
        grid.point_data["values"] = volume_data.flatten(order="F")

        # 添加轮廓（面渲染）
        contours = grid.contour(isosurfaces=8)
        if contours.n_points > 0:
            self._plotter.add_mesh(contours, opacity=0.5, color="white")

        # 添加体属性（体渲染）
        self._plotter.add_volume(grid, cmap="gray", opacity="sigmoid")

        self._plotter.reset_camera()
        self._plotter.show()

    def _load_dicom_volume(self, file_list: List[str]):
        """
        读取 DICOM 文件列表并构建三维体数据。

        Returns
        -------
        Tuple[np.ndarray, Tuple[float, float, float]] : (volume_array, spacing)
        """
        import numpy as np
        import pydicom

        # 读取所有文件的元数据并按 Z 轴排序
        slices = []
        for fp in file_list:
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=False)
                if hasattr(ds, "pixel_array"):
                    slices.append(ds)
            except Exception:
                continue

        if len(slices) < 2:
            raise ValueError("需要至少 2 张切片才能进行 3D 重建")

        # 按 ImagePositionPatient[2] 排序
        slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

        # 构建 3D 数组
        first = slices[0]
        pixel_spacing = first.PixelSpacing  # [row, col]
        slice_thickness = getattr(first, "SliceThickness", pixel_spacing[0])

        # 尝试从相邻切片计算真实的 Z 轴间距
        if len(slices) >= 2:
            z0 = float(slices[0].ImagePositionPatient[2])
            z1 = float(slices[1].ImagePositionPatient[2])
            slice_thickness = abs(z1 - z0)

        arr_shape = (len(slices), first.Rows, first.Columns)
        volume = np.zeros(arr_shape, dtype=np.float32)

        for i, ds in enumerate(slices):
            arr = ds.pixel_array.astype(np.float32)
            # 应用 rescale
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            arr = arr * slope + intercept
            volume[i] = arr

        spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))
        return volume, spacing
