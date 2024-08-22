---
sidebar_position: 4
---

# 🧹 Clearing data
In this section you will learn how to delete the app, etc.

## Flags
:::info
All `fguard data` stored at user local folder (different in many OS).
:::
- To remove cache run:
```bash
fguard remove --cache
```

- To remove configuration file run:
```bash
fguard remove --config
```

- To remove neural network model run:
```bash
fguard remove --model
```

## Deleting `fguard` (💀💀💀):
Execute previous two commands:
```bash
fguard remove --cache --config --model
```
After that remove python package using that:
```
pip uninstall fguard
```
**Forest Guard** was successfully deleted.