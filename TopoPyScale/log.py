"""
Messages of TopoPyScale, through the standard logging module (logger 'TopoPyScale', one child logger per module).

By default messages are shown in the terminal (or the notebook) as they happen, as print() did, so that a run can be
followed. Topoclass also writes them, with time stamps, to a log file in the outputs folder. Python warnings
(warnings.warn) raised during a Topoclass run go to the same places.

Usage:
    import TopoPyScale
    TopoPyScale.log.setup(verbose=False)                 # warnings and errors only
    TopoPyScale.log.setup(verbose='debug', logfile='run.log')

Config (Topoclass): project.verbose (True, False, or a level name) and project.log_file (True for
outputs/logs/topopyscale_<date>.log, a path, or False).

S. Filhol, J. Fiddes and contributors
"""
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger('TopoPyScale')
FILE_FORMAT = '%(asctime)s %(levelname)-7s %(name)s: %(message)s'


class _Terminal(logging.StreamHandler):
    """Handler writing to the current sys.stdout (also when it is redirected after the handler is created)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


class _TerminalFormat(logging.Formatter):
    """Messages as they are; warnings and errors prefixed by their level."""

    def format(self, record):
        message = super().format(record)
        return message if record.levelno < logging.WARNING else f'{record.levelname}: {message}'


def _level(verbose):
    if isinstance(verbose, str):
        level = logging.getLevelName(verbose.upper())
        if not isinstance(level, int):
            raise ValueError(f'verbose must be True, False or a logging level (debug, info, warning), not {verbose}')
        return level
    return logging.INFO if verbose else logging.WARNING


def setup(verbose=True, logfile=None, capture_warnings=False):
    """
    Where the messages of TopoPyScale go. Called with the defaults when TopoPyScale is imported.

    Args:
        verbose (bool or str): True: progress messages in the terminal (level INFO); False: warnings and errors only;
            or a level name ('debug', 'info', 'warning', 'error')
        logfile (str or Path): also write the messages (from INFO, or DEBUG if verbose='debug') to this file,
            appended, with time stamps. None: no file
        capture_warnings (bool): route Python warnings (warnings.warn) to the same terminal and file

    Returns:
        Path: the log file, or None
    """
    level = _level(verbose)
    for h in list(LOGGER.handlers):
        LOGGER.removeHandler(h)
        h.close()
    terminal = _Terminal()
    terminal.setLevel(level)
    terminal.setFormatter(_TerminalFormat('%(message)s'))
    handlers = [terminal]
    if logfile:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        f = logging.FileHandler(logfile)
        f.setLevel(min(level, logging.INFO))
        f.setFormatter(logging.Formatter(FILE_FORMAT))
        handlers.append(f)
    for h in handlers:
        LOGGER.addHandler(h)
    LOGGER.setLevel(min(h.level for h in handlers))
    LOGGER.propagate = False

    py_warnings = logging.getLogger('py.warnings')
    for h in list(py_warnings.handlers):
        if getattr(h, '_topopyscale', False):
            py_warnings.removeHandler(h)
    logging.captureWarnings(bool(capture_warnings))
    py_warnings.propagate = True
    if capture_warnings:
        for h in handlers:
            h._topopyscale = True
            py_warnings.addHandler(h)
        py_warnings.propagate = False
    return logfile


def get_logger(name):
    """Logger of a module: TopoPyScale.<module>."""
    return logging.getLogger(name if name.startswith('TopoPyScale') else f'TopoPyScale.{name}')
