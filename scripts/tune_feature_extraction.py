"""
GUI application to help manually tune LOAM feature extraction parameters

Author: Dan McGann
Date: May 2025
"""

import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(SCRIPT_DIR / ".." / "build" / "python"))

print(sys.path)
import loam


class FeatureViewer:
    """
    The visualizer for LOAM Features

    Based on: https://github.com/isl-org/Open3D/blob/main/examples/python/visualization/all_widgets.py
    """

    def __init__(self):
        # Define the Window
        self.window = gui.Application.instance.create_window(
            "Feature Viewer", 1280, 720
        )

        # The feature extraction parameters that available for edit
        self.fe_params = {
            "neighbor_points": 3,
            "number_sectors": 6,
            "max_edge_feats_per_sector": 10,
            "max_planar_feats_per_sector": 50,
            "edge_feat_threshold": 100.0,
            "planar_feat_threshold": 1.0,
            "occlusion_thresh": 0.5,
            "parallel_thresh": 1.0,
        }

        # The Lidar Parameters for the PCD
        self.lidar_params = {
            "rows": 64,
            "cols": 1024,
            "min_range": 1.0,
            "max_range": 100.0,
        }

        # The pointcloud to Evaluate
        self.pcd = None

        # The Materials for rendering
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.base_color = [0, 0, 0, 1]
        self.pcd_mat.point_size = 2

        self.plane_mat = rendering.MaterialRecord()
        self.plane_mat.base_color = [0, 0, 1, 1]
        self.plane_mat.point_size = 5

        self.edge_mat = rendering.MaterialRecord()
        self.edge_mat.base_color = [1, 0.8, 0, 1]
        self.edge_mat.point_size = 5

        # Define the Geometry Scene
        self._define_scene()

        # Define the Parameter selection Panel
        self._define_parameter_panel()

        # Add the Settings to the window
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._param_panel)

    def _define_scene(self):
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.set_background([1, 1, 1, 1])  # White Background
        self._scene.scene.scene.set_sun_light(
            [-1, -1, -1], [1, 1, 1], 100000
        )  # direction, color, intensity
        self._scene.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        self._scene.setup_camera(60, bbox, [0, 0, 0])

    def _add_lidar_param_setter(self, grid, param_name, data_type):
        layout = gui.Horiz(0.25 * self.window.theme.font_size)
        layout.add_child(gui.Label(f"{param_name}:"))
        editor = gui.NumberEdit(
            gui.NumberEdit.INT if data_type == "int" else gui.NumberEdit.DOUBLE
        )
        editor.double_value = self.lidar_params[param_name]
        editor.set_on_value_changed(lambda v: self.lidar_params.update({param_name: v}))
        layout.add_child(editor)
        grid.add_child(layout)

    def _add_parameter_tuner(self, param_name, data_type, limits):
        tuner = gui.VGrid(3, 0.25 * self.window.theme.font_size)
        tuner.add_child(gui.Label(f"{param_name}:"))
        # The slider for this parameter
        slider = gui.Slider(gui.Slider.INT if data_type == "int" else gui.Slider.DOUBLE)
        slider.set_limits(*limits)
        slider.double_value = self.fe_params[param_name]
        tuner.add_child(slider)
        # The number editor for this parameter
        editor = gui.NumberEdit(
            gui.NumberEdit.INT if data_type == "int" else gui.NumberEdit.DOUBLE
        )
        editor.set_limits(*limits)
        editor.double_value = self.fe_params[param_name]
        tuner.add_child(editor)

        # Setup the callbacks for this parameter
        slider.set_on_value_changed(
            lambda v: self._on_param_update(v, param_name, slider, editor)
        )
        editor.set_on_value_changed(
            lambda v: self._on_param_update(v, param_name, slider, editor)
        )

        # Add the parameter row to the layout
        self._param_panel.add_child(tuner)

    def _define_parameter_panel(self):
        em = self.window.theme.font_size
        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        self._param_panel = gui.Vert(
            0.25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em)
        )

        # Create a file-chooser widget for selecting the PCD file to run feature extraction on
        self._param_panel.add_child(gui.Label("Select PCD File to Evaluate:"))
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)

        # Create the horizontal widget for the row.
        fileedit_layout = gui.Horiz(0.5 * em)
        fileedit_layout.add_child(gui.Label("PCD File:"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)
        # add to the top-level (vertical) layout
        self._param_panel.add_child(fileedit_layout)

        self._param_panel.add_child(gui.Label("Specify Lidar Parameters:"))
        lidar_param_grid = gui.VGrid(2, 0.5 * em)
        self._add_lidar_param_setter(lidar_param_grid, "rows", "int")
        self._add_lidar_param_setter(lidar_param_grid, "min_range", "double")
        self._add_lidar_param_setter(lidar_param_grid, "cols", "int")
        self._add_lidar_param_setter(lidar_param_grid, "max_range", "double")
        self._param_panel.add_child(lidar_param_grid)

        # Add Tuners for all of the parameters
        self._param_panel.add_child(gui.Label("Adjust to Tune Features:"))
        self._add_parameter_tuner("neighbor_points", "int", (1, 20))
        self._add_parameter_tuner("number_sectors", "int", (1, 20))
        self._add_parameter_tuner("max_edge_feats_per_sector", "int", (1, 10000))
        self._add_parameter_tuner("max_planar_feats_per_sector", "int", (1, 10000))

        self._add_parameter_tuner("edge_feat_threshold", "double", (0.0, 1000.0))
        self._add_parameter_tuner("planar_feat_threshold", "double", (0.0, 1000.0))
        self._add_parameter_tuner("occlusion_thresh", "double", (0.0, 10.0))
        self._add_parameter_tuner("parallel_thresh", "double", (0.0, 10.0))

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.window.theme)
        filedlg.add_filter(".pcd", "Pointcloud (.pcd)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self.window.close_dialog()

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        self.pcd = o3d.io.read_point_cloud(path)
        self._scene.scene.remove_geometry("pcd")
        self._scene.scene.add_geometry("pcd", self.pcd, self.pcd_mat)
        self._update_features()
        self.window.close_dialog()

    def _on_param_update(self, new_val, param_name, slider, editor):
        self.fe_params[param_name] = new_val
        slider.double_value = new_val
        editor.double_value = slider.double_value  # slider val to clamp
        self._update_features()

    def _update_features(self):
        if self.pcd:
            fe_params = loam.FeatureExtractionParams()
            fe_params.neighbor_points = int(self.fe_params["neighbor_points"])
            fe_params.number_sectors = int(self.fe_params["number_sectors"])
            fe_params.max_edge_feats_per_sector = int(
                self.fe_params["max_edge_feats_per_sector"]
            )
            fe_params.max_planar_feats_per_sector = int(
                self.fe_params["max_planar_feats_per_sector"]
            )
            fe_params.edge_feat_threshold = self.fe_params["edge_feat_threshold"]
            fe_params.planar_feat_threshold = self.fe_params["planar_feat_threshold"]
            fe_params.occlusion_thresh = self.fe_params["occlusion_thresh"]
            fe_params.parallel_thresh = self.fe_params["parallel_thresh"]

            try:
                lidar_params = loam.LidarParams(
                    int(self.lidar_params["rows"]),
                    int(self.lidar_params["cols"]),
                    self.lidar_params["min_range"],
                    self.lidar_params["max_range"],
                )

                features = loam.extractFeatures(
                    np.asarray(self.pcd.points), lidar_params, fe_params
                )

                planar_features = o3d.geometry.PointCloud()
                planar_features.points = o3d.utility.Vector3dVector(
                    features.planar_points
                )
                edge_features = o3d.geometry.PointCloud()
                edge_features.points = o3d.utility.Vector3dVector(features.edge_points)

                self._scene.scene.remove_geometry("planar_features")
                self._scene.scene.add_geometry(
                    "planar_features", planar_features, self.plane_mat
                )
                self._scene.scene.remove_geometry("edge_features")
                self._scene.scene.add_geometry(
                    "edge_features", edge_features, self.edge_mat
                )
            except Exception:
                dlg = gui.Dialog("ERROR")
                em = self.window.theme.font_size
                dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
                dlg_layout.add_child(
                    gui.Label(
                        "Could not compute features on provided pointcloud. Check that the lidar parameters are correct."
                    )
                )
                ok_button = gui.Button("Ok")
                ok_button.set_on_clicked(self.window.close_dialog)
                dlg_layout.add_child(ok_button)
                dlg.add_child(dlg_layout)
                self.window.show_dialog(dlg)

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r

        param_sizes = self._param_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()
        )
        width = max(r.width * 0.35, 350)
        height = min(r.height, param_sizes.height)
        self._param_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)


def main():
    # We need to initialize the application, which finds the necessary shaders for
    # rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = FeatureViewer()

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
