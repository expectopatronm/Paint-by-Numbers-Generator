#!/usr/bin/env python3
import sys
import asyncio
from pyaxidraw import axidraw
from textual.app import App, ComposeResult
from textual.widgets import Button, Static
from textual.containers import Vertical

class PlotUI(App):
    BINDINGS = [
        ("p", "pause_resume", "Pause/Resume"),
        ("h", "walk_home", "Walk Home"),
        ("q", "quit_app", "Quit"),
    ]

    def __init__(self, svg_file: str, **kwargs):
        super().__init__(**kwargs)
        self.svg_file = svg_file
        self.ad = axidraw.AxiDraw()
        self.started = False
        self.paused = False
        self.current_svg = None

    def compose(self) -> ComposeResult:
        yield Static(f"File: {self.svg_file}")
        yield Static("Status: Not started", id="status")
        with Vertical():
            yield Button("Start", id="btnStart", variant="primary")
            yield Button("Pause/Resume", id="btnPause", disabled=True)
            yield Button("Walk Home", id="btnHome", disabled=True)
            yield Button("Quit", id="btnQuit", variant="error")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btnStart":
            await self.action_start_plot()
        elif btn_id == "btnPause":
            await self.action_pause_resume()
        elif btn_id == "btnHome":
            await self.action_walk_home()
        elif btn_id == "btnQuit":
            await self.action_quit_app()

    async def action_start_plot(self):
        if self.started:
            return
        self.started = True
        self.paused = False
        self.query_one("#status", Static).update("Status: Plotting…")
        self.query_one("#btnPause", Button).disabled = False
        self.query_one("#btnHome", Button).disabled = False

        # Setup machine and options
        self.ad.plot_setup(self.svg_file)

        MM_TO_INCH = 1.0/25.4
        self.ad.options.pen_pos_up     = 60
        self.ad.options.pen_pos_down   = 30
        self.ad.options.speed_pendown  = 75
        self.ad.options.speed_penup    = 75
        self.ad.options.accel          = 100
        self.ad.options.pen_rate_raise = 100
        self.ad.options.pen_rate_lower = 50
        self.ad.options.port           = "/dev/ttyACM0"
        self.ad.options.clip_to_page = True
        self.ad.options.auto_rotate = False  # disable auto-rotate so orientation is preserved
        self.ad.options.model = 2

        # Run the plot (in a thread so UI remains responsive)
        self.current_svg = await asyncio.to_thread(self.ad.plot_run, True)
        # At this point the plot either paused (by user/hardware) or finished
        self.paused = True
        self.query_one("#status", Static).update("Status: Paused or done")

    async def action_pause_resume(self):
        if not self.started:
            return
        btnPause = self.query_one("#btnPause", Button)
        if not self.paused:
            # Attempt to pause
            # Note: no documented ad.pause() method; user/hardware may need to press Pause
            self.query_one("#status", Static).update("Status: Waiting for pause (press machine pause)…")
            # We just set our internal flag; actual pause must be via hardware or layer-control
            self.paused = True
            btnPause.label = "Resume"
        else:
            # Resume running from saved SVG
            self.ad.plot_setup(self.current_svg)
            self.ad.options.mode = "res_plot"
            self.query_one("#status", Static).update("Status: Resuming…")
            self.current_svg = await asyncio.to_thread(self.ad.plot_run, True)
            self.query_one("#status", Static).update("Status: Paused or done")
            # Remain in paused/done state
            self.paused = True
            btnPause.label = "Pause"

    async def action_walk_home(self):
        if not self.started:
            return
        self.ad.plot_setup(self.current_svg)
        self.ad.options.mode = "res_home"
        self.query_one("#status", Static).update("Status: Returning home…")
        self.current_svg = await asyncio.to_thread(self.ad.plot_run, True)
        self.query_one("#status", Static).update("Status: Home reached")
        self.query_one("#btnPause", Button).disabled = True
        self.query_one("#btnHome", Button).disabled = True

    async def action_quit_app(self):
        if self.started:
            self.ad.disconnect()
        self.exit()

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <svg-file>")
        sys.exit(1)
    svg_file = sys.argv[1]
    app = PlotUI(svg_file=svg_file)
    app.run()

if __name__ == "__main__":
    main()
