#!/bin/bash

JOB_PARAMS=${1:-'--idx 0 --ishape 0 --stride 50'} # defaults to [0, 0, 50]

# SET PATHS HERE
FFMPEG_PATH=/opt/ffmpeg/
BLENDER_PATH=/usr/local/blender-2.79b/

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.79/python
export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.5:${BUNDLED_PYTHON}/lib/python3.5/site-packages
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

# FFMPEG
export LD_LIBRARY_PATH=${FFMPEG_PATH}/lib:${X264_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${FFMPEG_PATH}/bin:${PATH}


### RUN PART 1  --- Uses python3 because of Blender
$BLENDER_PATH/blender -b -t 1 -P Render_Animation.py -- ${JOB_PARAMS}

