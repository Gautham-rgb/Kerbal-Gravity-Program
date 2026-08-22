from __future__ import annotations

from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSlider
from PySide6.QtCore import Qt, QTimer
from trame.app import get_server, asynchronous
from trame.ui.vuetify3 import SinglePageLayout
from pyvista.trame.ui import plotter_ui

from basic_systems.builder import System
import pyvista as pv
import asyncio
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from basic_systems.renderer.renderer import SystemRenderer

class PySideVTKAdapter(QWidget):
    _SLIDER_STEPS = 1000

    def __init__(self, system: System, start_ut: float = 0.0, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.qt_interactor = QtInteractor(self)
        layout.addWidget(self.qt_interactor)

        self.renderer = SystemRenderer(system, start_ut=start_ut, plotter=self.qt_interactor, show_timeline_slider=False)
        self._timeline_max_ut = self.renderer.scene._compute_timeline_max_ut()

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, self._SLIDER_STEPS)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.timeline_slider)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _on_slider_changed(self, value: int):
        frac = value / self._SLIDER_STEPS
        self.renderer.updater.set_time(frac * self._timeline_max_ut)
        self.renderer.plotter.render()

    def _tick(self):
        # We use a fixed dt here for the timer-based update
        self.renderer.update(0.033)
        if not self.timeline_slider.isSliderDown():
            frac = max(0.0, min(1.0, self.renderer.curr_ut / self._timeline_max_ut))
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(int(frac * self._SLIDER_STEPS))
            self.timeline_slider.blockSignals(False)

    def closeEvent(self, event):
        self.timer.stop()
        self.renderer.plotter.close()
        event.accept()

class TrameWebAdapter:
    def __init__(self, system: System, start_ut: float = 0.0):
        pv.OFF_SCREEN = True
        self.renderer = SystemRenderer(system, start_ut=start_ut, headless=True)
        self.server = get_server()
        self.state, self.ctrl = self.server.state, self.server.controller#type: ignore
        self.view = None

        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("KRGP Web Render")
            with layout.content:
                self.view = plotter_ui(self.renderer.plotter)

        self.ctrl.on_server_ready.add(self._start_tick_loop)

    def _start_tick_loop(self, **kwargs):
        asynchronous.create_task(self._tick_loop())

    async def _tick_loop(self):
        while True:
            await asyncio.sleep(0.033)
            # Drive the real update path so tickets, HUD and positions advance
            # exactly like the desktop build.
            self.renderer.update(0.033)
            if self.view is not None:
                self.view.update()

    def start(self, port: int = 8080, open_browser: bool = False):
        self.server.start(port=port, open_browser=open_browser)#type: ignore
