#!/bin/bash

if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found, installing it now..."
    pip install gdown
fi

url1="https://drive.google.com/file/d/1-0WaAX0XgyA1IFYjEeO2aKUeMcl-1k7t/view?usp=drive_link"
url2="https://drive.google.com/file/d/1-5uE73rTGtVVokebogUjLWtsfd6nw1y0/view?usp=drive_link"

echo "Downloading default weight for faster rcnn..."
gdown "$url1"

echo "Downloading number plate weight for faster rcnn..."
gdown "$url2"

echo "Download complete."