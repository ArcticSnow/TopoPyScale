<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/log.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.log`
Messages of TopoPyScale, through the standard logging module (logger 'TopoPyScale', one child logger per module). 

By default messages are shown in the terminal (or the notebook) as they happen, as print() did, so that a run can be followed. Topoclass also writes them, with time stamps, to a log file in the outputs folder. Python warnings (warnings.warn) raised during a Topoclass run go to the same places. 

Usage:  import TopoPyScale  TopoPyScale.log.setup(verbose=False)                 # warnings and errors only  TopoPyScale.log.setup(verbose='debug', logfile='run.log') 

Config (Topoclass): project.verbose (True, False, or a level name) and project.log_file (True for outputs/logs/topopyscale_<date>.log, a path, or False). 

S. Filhol, J. Fiddes and contributors 

**Global Variables**
---------------
- **FILE_FORMAT**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/log.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `setup`

```python
setup(verbose=True, logfile=None, capture_warnings=False)
```

Where the messages of TopoPyScale go. Called with the defaults when TopoPyScale is imported. 



**Args:**
 
 - <b>`verbose`</b> (bool or str):  True: progress messages in the terminal (level INFO); False: warnings and errors only;  or a level name ('debug', 'info', 'warning', 'error') 
 - <b>`logfile`</b> (str or Path):  also write the messages (from INFO, or DEBUG if verbose='debug') to this file, 
 - <b>`appended, with time stamps. None`</b>:  no file 
 - <b>`capture_warnings`</b> (bool):  route Python warnings (warnings.warn) to the same terminal and file 



**Returns:**
 
 - <b>`Path`</b>:  the log file, or None 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/log.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_logger`

```python
get_logger(name)
```

Logger of a module: TopoPyScale.<module>. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
