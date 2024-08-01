#!/bin/bash

WORKSPACE=/workspace

docker run \
	--gpus all \
	-v $PWD:$WORKSPACE \
	-w $WORKSPACE \
	-it cupy/cupy \
	main.py $@
