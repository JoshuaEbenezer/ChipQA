#!/bin/sh
ffmpeg -y -i $1 -pix_fmt yuv420p -vf "minterpolate=fps=$3:mi_mode=dup" -an $2
