try:
    from .info import info
    from .conversion import conversion
    from .odinson import odinson
    from .spacy import spacy
    from .typing import typing
    from .processors import processors

    __version__ = info.version
except Exception as e:
    print(f"Failed to import modules: {e}")
