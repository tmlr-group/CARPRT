#!/bin/bash

CUDA_VISIBLE_DEVICES=0




python test.py --config configs \
                --datasets caltech101/dtd/eurosat/food101/oxford_pets \
                --backbone ViT-B/16 