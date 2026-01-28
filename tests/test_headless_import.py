import os

def test_import_viewer_headless():
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import pygame  # noqa: F401
    from viewers.app import App  # noqa: F401
