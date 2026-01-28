# Centralized keymap for all hotkeys and help overlay generation

KEYMAP = [
    {"keys": "Ctrl+S", "action": "Save Q-table", "scene": "Simulation", "desc": "Save the current Q-table to file"},
    {"keys": "Ctrl+L", "action": "Load Q-table", "scene": "Simulation", "desc": "Load a Q-table from file"},
    {"keys": "Ctrl+E", "action": "Export Screenshot", "scene": "Simulation", "desc": "Export a screenshot of the viewer"},
    {"keys": "Ctrl+O", "action": "Save Env Snapshot", "scene": "Simulation", "desc": "Save the current environment state"},
    {"keys": "Ctrl+I", "action": "Load Env Snapshot", "scene": "Simulation", "desc": "Load an environment state from file"},
    {"keys": "Ctrl+X", "action": "Export Stats", "scene": "Simulation", "desc": "Export run statistics as JSON/CSV"},
    {"keys": "Ctrl+T", "action": "Toggle Telemetry Overlay", "scene": "Simulation", "desc": "Show/hide telemetry overlay"},
    {"keys": "Ctrl+K", "action": "Toggle Run History", "scene": "Simulation", "desc": "Show/hide recent episode history in the tool sidebar"},
    {"keys": "Space", "action": "Pause/Resume", "scene": "Simulation", "desc": "Pause or resume simulation"},
    {"keys": ".", "action": "Step Once", "scene": "Simulation", "desc": "Advance simulation by one step"},
    {"keys": "R", "action": "Reset Episode", "scene": "Simulation", "desc": "Restart the current episode"},
    {"keys": "M", "action": "Toggle Policy", "scene": "Simulation", "desc": "Switch between greedy/epsilon policy"},
    {"keys": "H", "action": "Toggle Heatmap", "scene": "Simulation", "desc": "Show/hide Q-value heatmap"},
    {"keys": "P", "action": "Toggle Policy Overlay", "scene": "Simulation", "desc": "Show/hide policy arrows"},
    {"keys": "Q", "action": "Toggle Q-hover Panel", "scene": "Simulation", "desc": "Show/hide Q-value hover panel"},
    {"keys": "D", "action": "Toggle Debug Overlay", "scene": "Simulation", "desc": "Show/hide debug overlay"},
    {"keys": "?", "action": "Help Overlay", "scene": "Simulation", "desc": "Show/hide help overlay"},
    {"keys": "Esc", "action": "Back/Exit", "scene": "All", "desc": "Go back or exit current screen"},
]

def get_keymap(scene=None):
    if scene:
        return [k for k in KEYMAP if k["scene"] == scene or k["scene"] == "All"]
    return KEYMAP
