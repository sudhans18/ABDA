"""
ABDA Phase 1 — Integration package.

Public API (imported lazily to avoid runpy sys.modules warnings when
running py -m integration.build_session directly):

  build_session          — assemble one MultimodalRecord
  validate_alignment     — run integration validation
  MultimodalRecord       — top-level schema class
  record_to_display_dict — display-ready dict for UI/API

All symbols are importable directly from integration.schema,
integration.build_session, integration.validate_alignment, and
integration.visualize as well.
"""


def __getattr__(name):
    """Lazy-load top-level integration symbols to avoid double-import warnings."""
    _schema_names = {
        "MultimodalRecord", "SegmentIdentity", "TextRecord", "SpeechRecord",
        "SpeechQuality", "FacialRecord", "OpenFaceStatus", "IntegrationSummary",
    }
    _build_names = {"build_session", "retrieve_text_record", "retrieve_speech_record", "retrieve_facial_record"}
    _validate_names = {"validate_alignment"}
    _visualize_names = {"record_to_display_dict"}

    if name in _schema_names:
        from . import schema as _schema
        return getattr(_schema, name)
    if name in _build_names:
        from . import build_session as _bs
        return getattr(_bs, name)
    if name in _validate_names:
        from . import validate_alignment as _va
        return getattr(_va, name)
    if name in _visualize_names:
        from . import visualize as _viz
        return getattr(_viz, name)
    raise AttributeError(f"module 'integration' has no attribute {name!r}")


__all__ = [
    "build_session",
    "retrieve_text_record",
    "retrieve_speech_record",
    "retrieve_facial_record",
    "validate_alignment",
    "record_to_display_dict",
    "MultimodalRecord",
    "SegmentIdentity",
    "TextRecord",
    "SpeechRecord",
    "SpeechQuality",
    "FacialRecord",
    "OpenFaceStatus",
    "IntegrationSummary",
]
