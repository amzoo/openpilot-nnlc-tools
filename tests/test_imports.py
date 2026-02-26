"""Smoke tests: verify all nnlc_tools modules import successfully."""


def test_import_init():
    import nnlc_tools  # noqa: F401


def test_import_logreader():
    from nnlc_tools.logreader import LogReader  # noqa: F401


def test_import_extract():
    import nnlc_tools.extract_lateral_data  # noqa: F401


def test_import_score():
    import nnlc_tools.score_routes  # noqa: F401


def test_import_visualize():
    import nnlc_tools.visualize_coverage  # noqa: F401


def test_import_sync():
    import nnlc_tools.sync_rlogs  # noqa: F401
