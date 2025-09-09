import traceback
import datetime
import os.path

import torch
import numpy as np
import viser
import viser.transforms as vtf
import re


class ScPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        self.server = server
        self.viewer = viewer
        self.tab = tab

        self._setup_traj_folder()
        self._setup_refine_folder()
        self._setup_vis_folder()

    def _setup_traj_folder(self):
        with self.server.gui.add_folder("Traj"):
            dummy_button = self.server.gui.add_button("Dummy")
        
    def _setup_refine_folder(self):
        with self.server.gui.add_folder("Refine"):
            dummy_button = self.server.gui.add_button("Dummy")

    def _setup_vis_folder(self):
        with self.server.gui.add_folder("Visualization"):
            dummy_button = self.server.gui.add_button("Dummy")