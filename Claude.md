New repo ~/OverGaze (GitHub ahmadSum1/OverGaze, published on GitHub Pages like MaskFlow ( https://ahmadsum1.github.io/MaskFlow/ ) ).
Starting point: `/home/skb/gazeflow`, a static browser app. It is a first draft: for me nothing loads or works.

Goal: a working tool that overlays eye-tracking gaze / fixations on a video (local file or direct link), turns them
into saliency maps with settings I can change live, and exports the maps.
- 360° equirectangular video (Gaussian on the sphere) and plain 2D video
- settings: Gaussian size in degrees, pooling window ± t seconds (flat / fading weights, past-only option),
  motion compensation (optical flow, so pooled gaze follows the moving camera or object), speed filter,
  rotation and flips, display options
- coverage readout: share of the frame ≥ 0.1 of the frame maximum
- export: maps as PNG or .npy with the settings saved alongside; overlay video and snapshot
- must read my real data: VR-EyeTracking and YT360-EyeTracking gaze files and their videos
  (paths in the video-sal-data README); open one of each and confirm the overlay lands on what people look at

Steps:
1. Serve it locally, open it in Chrome, reproduce my "nothing works" and fix it. Check the console.
2. Review the draft and keep, fix or rewrite whatever you think best; match MaskFlow's look.
3. Check with the bundled sample, then the real clips: a VR-EyeTracking clip with σ 3°, ±1 s and motion compensation
   should give coverage close to the data agent's mc3 maps (about 3.35 %).
4. README, tests, a short screenshot per feature. Commit; don't push.
Plain words in your report, no unexplained short forms.

Dataset: Sample at `/home/skb/datasets/video_sal/yt360` `/home/skb/datasets/video_sal/ vr_eyetracking` `/home/skb/dataset_checkpoints/YT360-EyeTracking_Dataset` `/home/skb/dataset_checkpoints/Salient360`