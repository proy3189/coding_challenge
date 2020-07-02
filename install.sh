#!/bin/bash

source activate talpaenv
python setup.py sdist
pip install ./dist/talpa-1.0.tar.gz
