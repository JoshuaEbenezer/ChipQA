#!/bin/sh
ffmpeg -y -i $1 -vf scale=3840x2160:flags=fast_bilinear $2 
