# CSCI 1470/2470 Final Project
## Contributors
Jorge Isaac Chang Ortega: jchang88

Eric Wang: ewang34 	

Ruotao Zhang: rzhang63

## File Summary
count_packages.py: Main script for running the package counting. Loads in model trained by train_fasterRCNN.py. Implements package counting with object tracking by utilizing sort.py.

sort.py: SORT object tracking API implemented by Bewley et al. Uses filterpy for Kalman Filter implementation. 

@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}

train_fasterRCNN.py: trains the Faster RCNN network

video2frames.py: helper script for splitting videos into frames for labeling.


