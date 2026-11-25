# OpenCV.js 4.10.0

`opencv.js` is the official OpenCV.js build of OpenCV 4.10.0 (https://docs.opencv.org/4.10.0/opencv.js), unchanged,
under the Apache 2.0 licence in `LICENSE`. OverGaze uses its Farnebäck dense optical flow for "Analyse motion"
(`src/flow-worker.js` loads it the first time it is needed); it is served from this repository, so no other site is
contacted at run time.
