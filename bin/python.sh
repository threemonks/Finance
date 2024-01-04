#! /bin/bash

# add relative python library path
PYLIBDIR=`/bin/readlink -f $(dirname $0)/../`
PYTHONPATH=$PYTHONPATH:$PYLIBDIR
export PYTHONPATH

DIR=`/bin/readlink -f $(dirname $0)`
BASEDIR=${DIR%/bin}

# launch the python script via real python interpreter
source ./venv/bin/activate
cd $BASEDIR && ./venv/bin/python "$@"
