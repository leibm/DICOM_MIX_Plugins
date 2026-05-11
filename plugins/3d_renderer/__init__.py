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
    "version": "1.0.1",
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
        if self._window is not None:
            try:
                self._window.close()
            except Exception:
                pass
            self._window = None

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
        """创建并显示 3D 渲染窗口，带 MPR 软件渲染回退。"""
        import numpy as np

        # 读取并排序 DICOM 文件
        try:
            volume_data, spacing = self._load_dicom_volume(file_list)
        except Exception as e:
            self._show_message(f"加载 DICOM 数据失败:\n{e}")
            return

        # 先尝试硬件加速的 3D 渲染
        use_hardware = self._try_hardware_rendering(volume_data, spacing)
        if not use_hardware:
            # 回退到纯软件的 MPR 2D 切片浏览
            self._show_mpr_window(volume_data, spacing)

    def _try_hardware_rendering(self, volume_data, spacing) -> bool:
        """尝试使用 PyVistaQt 进行硬件 3D 渲染。返回是否成功。"""
        try:
            import pyvista as pv
            import vtk
            from pyvistaqt import QtInteractor

            # 禁用多重采样，避免部分显卡驱动出现 pixel format 错误
            vtk.vtkOpenGLRenderWindow.SetGlobalMaximumNumberOfMultiSamples(0)

            self._plotter = QtInteractor()

            grid = pv.ImageData()
            grid.dimensions = volume_data.shape
            grid.spacing = spacing
            grid.origin = (0, 0, 0)
            grid.point_data["values"] = volume_data.flatten(order="F")

            contours = grid.contour(isosurfaces=8)
            if contours.n_points > 0:
                self._plotter.add_mesh(contours, opacity=0.5, color="white")

            self._plotter.add_volume(grid, cmap="gray", opacity="sigmoid")
            self._plotter.reset_camera()
            self._plotter.show()
            return True

        except Exception as e:
            logger.warning(f"硬件 3D 渲染失败，将回退到 MPR: {e}")
            self._plotter = None
            return False

    def _show_mpr_window(self, volume_data: "np.ndarray", spacing: tuple):
        """纯软件 MPR 多平面重建窗口（无 OpenGL 依赖）。"""
        import numpy as np
        from PySide6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
            QGridLayout, QGroupBox, QApplication
        )
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap

        class MPRViewer(QWidget):
            def __init__(self, volume, spacing, parent=None):
                super().__init__(parent)
                self.volume = volume
                self.spacing = spacing
                self.nz, self.ny, self.nx = volume.shape

                # 计算窗宽窗位默认值
                self.vmin = float(volume.min())
                self.vmax = float(volume.max())

                self.setWindowTitle("MPR 多平面重建")
                self.resize(1200, 800)

                main_layout = QHBoxLayout(self)

                # 三个视图组
                self.axial_group = self._create_view_group("轴位 (Axial)", self.nz, self._update_axial)
                self.coronal_group = self._create_view_group("冠状位 (Coronal)", self.ny, self._update_coronal)
                self.sagittal_group = self._create_view_group("矢状位 (Sagittal)", self.nx, self._update_sagittal)

                main_layout.addWidget(self.axial_group)
                main_layout.addWidget(self.coronal_group)
                main_layout.addWidget(self.sagittal_group)

                # 初始显示中间切片
                self.axial_slider.setValue(self.nz // 2)
                self.coronal_slider.setValue(self.ny // 2)
                self.sagittal_slider.setValue(self.nx // 2)

            def _create_view_group(self, title, max_val, update_func):
                group = QGroupBox(title)
                layout = QVBoxLayout(group)

                label = QLabel()
                label.setMinimumSize(350, 350)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("background-color: black;")
                layout.addWidget(label)

                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, max_val - 1)
                slider.valueChanged.connect(update_func)
                layout.addWidget(slider)

                info = QLabel(f"切片: 0 / {max_val}")
                layout.addWidget(info)

                setattr(self, title.split()[0].lower() + "_label", label)
                setattr(self, title.split()[0].lower() + "_slider", slider)
                setattr(self, title.split()[0].lower() + "_info", info)
                return group

            def _array_to_pixmap(self, arr_2d):
                """将 2D numpy 数组转为 QPixmap。"""
                # 窗宽窗位映射到 0-255
                arr = np.clip((arr_2d - self.vmin) / (self.vmax - self.vmin) * 255, 0, 255).astype(np.uint8)
                h, w = arr.shape
                image = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(image)
                return pixmap

            def _update_axial(self, z):
                slice_data = self.volume[z, :, :]
                pixmap = self._array_to_pixmap(slice_data)
                label = self.axial_label
                label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.axial_info.setText(f"切片: {z + 1} / {self.nz}")

            def _update_coronal(self, y):
                slice_data = self.volume[:, y, :]
                pixmap = self._array_to_pixmap(slice_data)
                label = self.coronal_label
                label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.coronal_info.setText(f"切片: {y + 1} / {self.ny}")

            def _update_sagittal(self, x):
                slice_data = self.volume[:, :, x]
                pixmap = self._array_to_pixmap(slice_data)
                label = self.sagittal_label
                label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.sagittal_info.setText(f"切片: {x + 1} / {self.nx}")

            def resizeEvent(self, event):
                super().resizeEvent(event)
                # 窗口大小改变时重绘当前切片
                self._update_axial(self.axial_slider.value())
                self._update_coronal(self.coronal_slider.value())
                self._update_sagittal(self.sagittal_slider.value())

        self._window = MPRViewer(volume_data, spacing)
        self._window.show()

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
