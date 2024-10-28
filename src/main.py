from ui import MusicProductionUI
from nicegui import ui

if __name__ in {"__main__", "__mp_main__"}:
    
    app = MusicProductionUI()
    ui.run(title='Music Production Assistant', host='0.0.0.0', port=8001)
