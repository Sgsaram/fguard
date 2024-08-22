---
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 📦 Installation

We are using python packaging system for greater convenience. Let's install Forest Guard CLI on you PC.

## Dependencies

First you must have Python 3 installed on your operating system. If you don't,
**[download it](https://www.python.org/downloads/)**.
Also make sure you have `pip` ready to use (python package manager).

To check that everything works, run:
```bash
python --version
pip --version
```

## Python package

That command will install the package **globally** on your computer:
```bash
pip install fguard
```
Forest Guard now is ready to use!

:::tip

You can also use **local python virtual environment**.

- Select a folder where you want to work (let be the place where folder stored `path-to-folder`)
- Go to that directory. You can use this command:
```bash
cd "<path/to/folder>"
```
- Run:
```bash
python -m venv venv
```
It will create environment in `path-to-folder/venv`

- Now activate it (also you need to do it everytime you are using the app):

<Tabs>
    <TabItem value="apple" label="Windows" default>
    ```bash
    venv\Scripts\activate
    ```
  </TabItem>
  <TabItem value="orange" label="Linux/MacOS">
    ```bash
    source venv/bin/activate
    ```
  </TabItem>
</Tabs>

- Now install `fguard` as we did before (now it will install package **locally**).

When you finish your work, you need to deactivate environment:
```bash
deactivate
```

:::
