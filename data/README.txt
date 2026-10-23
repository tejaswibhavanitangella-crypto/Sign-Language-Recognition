DATASET SETUP

1. Download the ISL dataset from Kaggle:
   https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl

2. Put the extracted class folders inside this folder:

   data/dataset/A/
   data/dataset/B/
   ...

The full dataset can remain here. You do NOT need to manually delete images.
The training program automatically selects a maximum of 200 images from each
class, using a fixed random seed so the selection is reproducible.

STATIC-ONLY PROJECT

The default training command excludes J and Z because these ISL alphabet signs
involve movement. If your project guide specifically wants those classes, you
can include them with:

   python src/train.py --exclude

To use all images from every included class instead of the 200-per-class limit:

   python src/train.py --images-per-class 0
