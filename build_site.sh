#!/bin/sh
# Puts the web page, the Python code and the models together in _site/ (what GitHub Pages serves).
set -e
rm -rf _site
mkdir -p _site/models
cp -r web/. _site/
cp attendance.py register.py _site/
cp models/face_detection_yunet_2023mar.onnx models/face_recognition_sface_2021dec_int8.onnx _site/models/
touch _site/.nojekyll
