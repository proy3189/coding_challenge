#!/bin/bash

source activate automlenv
python setup.py sdist
pip install ./dist/talpa-1.0.tar.gz
