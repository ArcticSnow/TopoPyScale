<!-- markdownlint-disable -->

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/config_check.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `TopoPyScale.config_check`
Check a config.yml against the keys TopoPyScale reads, listed in config_reference.yml (the example of the documentation). Keys of older versions are reported with what replaced them, unknown keys with the closest documented key, as a misspelled key would otherwise be read as empty without notice. 

S. Filhol, J. Fiddes and contributors 

**Global Variables**
---------------
- **DATA_KEYS**
- **OLD_KEYS**

---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/config_check.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `flatten`

```python
flatten(d, prefix='')
```

Dotted keys of a nested dictionary, stopping at DATA_KEYS. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/config_check.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reference_keys`

```python
reference_keys()
```

Keys of config_reference.yml. 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/config_check.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_config`

```python
check_config(config)
```

Messages about the keys of a configuration that TopoPyScale does not read. 



**Args:**
 
 - <b>`config`</b> (dict):  configuration as read from config.yml 



**Returns:**
 
 - <b>`list`</b>:  one message per key: renamed or removed key, or unknown key with the closest documented key 


---

<a href="https://github.com/ArcticSnow/TopoPyScale/TopoPyScale/config_check.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `warn_config`

```python
warn_config(config)
```

Log check_config() messages as warnings. Returns the messages. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
