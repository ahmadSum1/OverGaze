# Validation

How OverGaze's maps compare with published dataset maps, and which ways of building a map from gaze predict where other people look. Numbers only: the dataset videos have no licence for redistribution, so no frames or maps from them are in this repository. The screenshots in `docs/screenshots` use the synthetic samples.

Contents: 1. reproducing the published training maps · 2. the Salient360 test set and the measures · 3. motion estimates · 4. pooling, motion and a learned predictor (report 3) · 5. weighting the current frame and scene memory (report 4) · 6. comparison fixations from other videos (report 5) · 7. training videos · 8. earlier reports that were replaced · 9. faults found along the way.

## 1. Reproducing the published training maps

Settings in OverGaze: σ 3°, window ±1 s, Gaussian weights (σ = 0.5 s), speed filter 75°/s (VR-EyeTracking), motion analysed at 256 × 128 with the "play" method, motion compensation on. The app's raw maps (512 × 256) were averaged to 64 × 128 blocks, as in the reference maps ("mc3", `maps_mc3_*`, motion-compensated pooling by the data agent), and compared on 40–117 frames per clip. Coverage = share of the frame at or above 0.1 of the frame maximum, per pixel, mean over the sampled frames. Correlation of app map and reference map.

| clip | coverage app | coverage reference | correlation with the published map |
|---|---|---|---|
| VR-EyeTracking 001 | 6.20 % | 6.33 % | 0.985 |
| VR-EyeTracking 002 | 2.23 % | 2.29 % | 0.947 |
| VR-EyeTracking 092 | 2.62 % | 2.68 % | 0.994 |
| VR-EyeTracking 117 | 1.69 % | 1.43 % | 0.937 |
| VR-EyeTracking 050 (60 frames per second, reference uses every 2nd frame) | 6.65 % | 6.43 % | 0.950 |
| YT360 `-ZrvrKMC7Tc-28` (6 conditions pooled, 29.97 frames per second) | 7.47 % | 7.46 % | 0.945 |

Over all 132 reference VR-EyeTracking clips the mean coverage is 3.43 % (per pixel, not weighted by area); the five clips above average 3.8 % in both app and reference. On YT360 the correlation is highest at lag 0 (0.945; 0.87 at ±10 frames), so time alignment, orientation and the clip-start offset are right. A 25 frames per second YT360 clip (`pyRnPVbCmII-543`) is measured as 25 frames per second.

This table was made with OverGaze's built-in motion estimate before it was fixed (section 3). Section 7 repeats the check on 19 training videos with OpenCV motion.

**Conclusion.** OverGaze rebuilds the published "mc3" training maps (σ 3°, ±1 s with Gaussian weights over time, motion compensation) closely: correlation 0.94–0.99 on these clips, and 0.99 on average over the 19 training videos of section 7, where the same recipe was rebuilt with OpenCV motion.

## 2. The Salient360 test set and the measures

**Data.** Salient360 video set, all 19 videos, head and eye fixations of both eyes. Every 5th frame, leaving out the first and last second: 1887 frames. Each video at its true frame rate (read from the file header; 23.976 to 30 frames per second). The dataset's own map of each frame (the `.bin` file, one frame later in its numbering) is the reference for "kept" and "added". Test clips of the predictor in the tables below: 1_PortoRiverside, 4_Ocean, 7_GazaFishermen.

**Maps.** All at 256 × 128 (equirectangular), with a Gaussian of σ 2° on the sphere per gaze point, divided by the frame's maximum.

- **this frame**: the fixations active during the frame.
- **this frame, wider Gaussian**: this frame, with σ enlarged until its salient area equals that of "pooled ±1 s" on the same frame (bisection between 2° and 60°): a wider blob instead of pooling over time.
- **pooled ±1 s**: gaze from 1 s before to 1 s after, weighted over time by a Gaussian with σ 0.5 s; the reference for the significance tests.
- **past 1 s**: the same, past only, as a live mask would have it.
- **… + motion**: each pooled gaze point moved along the motion to where its content is at this frame. Motion = OpenCV dense optical flow (DIS, "dense inverse search") at 256 × 128, 10 fields per second, made with `tools/motion_opencv.py` and loaded with "Load motion file".
- **this frame + pooled**: "this frame" and "pooled ±1 s", each divided by its maximum, added, divided by the maximum again.
- **pooled, peaked τ = …**: ±1 s, each moment weighted exp(−|Δt| / τ): 1 at the current frame, 0.37 at τ seconds away. "past 1 s, peaked" uses the past half only.
- **pooled + memory w = …**: "pooled, peaked τ = 0.25 s + motion" plus w × the memory map, then divided by the maximum. The memory map is everything people looked at since the clip started (past only), moved along the motion to the current frame and divided by its own maximum. "past 1 s + memory" starts from "past 1 s, peaked τ = 0.25 s + motion" instead, so it works live. τ = 0.25 s was chosen before any result was known.
- **SalFormer360-ft**: SalFormer360 fine-tuned on 60 Salient360 still images (yolo-sal, `run_salformer360_ft.sh`, seed 1), its maps as produced, resized and divided by the maximum. None of the 19 videos was in its training data.
- **average fixation map of this video**: control: all fixations of one half of the people over the whole video, in one map.

**Measures.**

- **salient area**: where a map is at least 0.1 of its maximum, measured on the sphere (each pixel weighted by its area).
- **kept**: share of the dataset map's salient area that is also salient in ours (1 = all of it).
- **added**: our salient area outside the dataset map's, as a multiple of the dataset map's area (summed over frames before dividing). Not a score on its own.
- **held out**: maps built from one half of the people, scored on the other half's fixations in the same frame. 5 random splits × both directions; both eyes of a person always in the same half (left and right eye recordings paired by matching fixation times and positions, about 50 pairs per video); frames where either half has no fixation are left out.
- **held-out hit rate**: share of the other half's fixations inside our salient area.
- **hits per added area, × chance**: the share of the other half's fixations that fall in the area our map adds beyond half 1's "this frame" area, divided by that area's share of the sphere, relative to an area of that size placed at random outside half 1's area.
- **AUC-Judd**: use the map's values under the fixations as thresholds; for each, the share of the other half's fixations at or above it against the share of the sphere at or above it; the area under that curve. 0.5 = no better than random, 1 = perfect. Fair to wide maps.
- **shuffled AUC, within video**: the chance that a held-out fixation gets a higher map value than a comparison fixation from another moment of the same video (more than 2 s away). It cancels what is common to the whole video: the bias towards the equator and also the video's static salient objects.
- **shuffled AUC, across videos**: the same with comparison fixations from the other 18 videos of the set (all their fixations, any time). It cancels the bias towards the equator and other habits common to all videos, but not a video's own salient objects (section 6).
- **NSS**: the map rescaled to mean 0 and spread 1 over the sphere, averaged at the other half's fixations: how many spreads above average the map is where people look. 0 = no better than random.
- **significance**: per video, the mean difference from "pooled ±1 s" over its frames, with a 95 % interval from a block bootstrap (blocks of 5 evaluated frames, about 1 s, 2,000 resamples; neighbouring frames are not independent). Over videos: how many are better and worse, how many have the whole interval above or below 0, and a two-sided sign test on better against worse.

## 3. Motion estimates

The motion field that OverGaze used at 16_Turtle frame 435 (frames 434 → 437, grabbed exactly) showed movement on the dry sand next to the surf. OpenCV on the same two frames showed none: median movement where the two frames do not change was 0.61 px for OverGaze's own estimate and 0.04 px for OpenCV (Farnebäck and DIS). The frame grabbing was right; the estimate spread the surf's motion into still areas. The built-in estimate was then fixed (pull request #8) and OpenCV motion files became loadable. Over 19 videos, about 200 frame pairs each, 0.1 s apart, 256 × 128, estimates run outside the browser on frames decoded by OpenCV:

| estimate | movement where the frames do not change, median per pair, mean (px) | worst pair, mean over videos (px) | pairs over 0.5 px | difference after moving the second frame back (0–255) | distance from DIS where DIS sees textured movement (px) |
|---|---:|---:|---:|---:|---:|
| OpenCV DIS | 0.30 | 1.30 | 586 of 3792 | 1.60 | — |
| built-in, before the fix | 0.52 | 2.42 | 923 of 3792 | 1.81 | 1.88 |
| built-in, after the fix | 0.28 | 1.20 | 600 of 3792 | 1.27 | 1.27 |

With no movement at all, the difference is 2.66. On frames moved by a known 1–6 px the median error was 0.06 px (DIS), 0.26 px (before the fix) and 0.11 px (after); on the bundled sample's known camera pan 0.12, 0.27 and 0.15 px. On 6_DroneFlight (moving camera) the fixed estimate calls 6.9 % of the pixels that DIS sees moving more than 1 px still.

**Conclusion (round 1).** The fixed built-in estimate no longer invents movement in still areas and is as good as DIS there, but on moving texture such as surf it is still about 1.3 px from DIS on average. On 16_Turtle frame 440, motion compensation with the fixed estimate keeps 67.6 % of the dataset map, against 95.2 % with DIS.

### Round 2: OpenCV Farnebäck in the browser, with checks

On a fixed-camera YT360 clip (Lc5xpQBDWyg-69 at 72 s) the built-in estimate still drew long arrows in the sky. OpenCV.js 4.10 (served from the repository, loaded when "Analyse motion" is pressed) provides Farnebäck's method; DIS is not in OpenCV.js. Three checks were added to the estimate in the browser: the motion back from the second frame must cancel the motion forward (else 0); textured patches (5 × 5, mean step to the next pixel above 8 of 255) that changed by less than 3 of 255 get no motion; rows beyond ±70° latitude take the horizontal motion of the last trusted row. Compared on three clips (frames decoded by OpenCV at 256 × 128, 0.1 s apart; known shifts: 15 frames of each clip moved by 1–6 px across and up to 2 px down, with noise):

| estimate | still pixels, 16_Turtle (median / 95th percentile, px) | still pixels, Lc5xpQBDWyg-69 | still pixels, 6_DroneFlight (moving camera) | known shifts, 16_Turtle (error, px: whole frame / between ±70°) | known shifts, Lc5xpQBDWyg-69 | known shifts, 6_DroneFlight | computing per minute of video (10 fields per second, one core) |
|---|---:|---:|---:|---:|---:|---:|---:|
| built-in before the first fix | 0.09 / 1.95 | 0.06 / 0.69 | 0.63 / 3.67 | 0.20 / 0.20 | 0.27 / 0.24 | 0.33 / 0.34 | 43 s |
| built-in | 0.04 / 0.84 | 0.02 / 0.30 | 0.42 / 2.43 | 0.10 / 0.10 | 0.13 / 0.12 | 0.14 / 0.13 | 86 s |
| built-in + checks | 0.00 / 0.28 | 0.00 / 0.11 | 0.20 / 1.54 | 0.11 / 0.10 | 0.14 / 0.12 | 0.16 / 0.13 | 176 s |
| Farnebäck | 0.02 / 0.60 | 0.02 / 0.13 | 0.02 / 0.74 | 0.06 / 0.06 | 0.05 / 0.05 | 0.08 / 0.07 | 15 s |
| Farnebäck + checks (now the default) | 0.00 / 0.37 | 0.00 / 0.07 | 0.00 / 0.67 | 0.08 / 0.06 | 0.05 / 0.05 | 0.66 / 0.07 | 29 s |
| OpenCV DIS (motion file) | 0.03 / 0.46 | 0.02 / 0.14 | 0.38 / 1.62 | 0.06 / 0.06 | 0.07 / 0.06 | 0.07 / 0.07 | — |

Whole analysis in the browser on 16_Turtle (20 s, 4K): with "seek" grabbing about 8.6 minutes per minute of video for every estimate (seeking in the 4K video takes most of the time); with "play" grabbing it runs at the speed of the video, and Farnebäck with checks kept up (195 of 200 fields) while the built-in estimate with checks did not (73 of 200). OpenCV.js took under a second to start once loaded. The DIS tool (`tools/motion_opencv.py`) took 6.2 s for the 20 s clip, including decoding. Still pixels = pixels whose brightness changed by at most 1 of 255 between the two frames; on the moving-camera clip some of them really move (flat areas), which is why DIS shows motion there. The checks set textured still pixels to 0 by design, so their still-pixel numbers are not a fair contest; the known shifts are. The larger whole-frame error of "Farnebäck + checks" on 6_DroneFlight comes from the pole rows: a uniform shift downwards is not a motion a 360° camera makes near the poles, and those rows get no vertical motion.

**Salient360 scores with the new browser estimate** (1887 frames, held out as in section 2; motion files made by running the browser code outside the browser on the frames OpenCV decoded):

| map | kept | held-out hit rate | AUC-Judd | shuffled AUC, within video | shuffled AUC, across videos | NSS |
|---|---:|---:|---:|---:|---:|---:|
| pooled ±1 s | 0.580 | 0.356 | 0.807 | 0.574 | 0.730 | 3.29 |
| pooled ±1 s + motion (OpenCV DIS file) | 0.578 | 0.362 | 0.810 | 0.575 | 0.732 | 3.30 |
| pooled ±1 s + motion (browser Farnebäck with checks) | 0.575 | 0.362 | 0.810 | 0.576 | 0.732 | 3.30 |

| measure | Farnebäck − DIS, mean per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| held-out hit rate | -0.0006 | 11 / 8 | 4 / 3 | 0.648 |
| AUC-Judd | +0.0000 | 9 / 10 | 2 / 2 | 1.000 |
| shuffled AUC, within video | +0.0004 | 15 / 4 | 3 / 1 | 0.019 |
| shuffled AUC, across videos | +0.0003 | 13 / 6 | 2 / 1 | 0.167 |
| NSS | +0.0012 | 10 / 9 | 2 / 2 | 1.000 |

**Conclusion (round 2).** Farnebäck in the browser, with the checks, is now the default estimate. Its errors on known shifts (0.05–0.08 px away from the poles) match DIS (0.06–0.07), it barely moves still pixels, it keeps up with real-time playback, and the Salient360 scores with it equal those with DIS motion files to three decimals (the DIS row reproduces section 4). On 16_Turtle frame 440 it keeps 89.2 % of the dataset map (DIS 95.2 %, no motion 90.4 %, built-in 67.6 %). Its weaknesses: flat, heavily compressed areas under a turning camera (the bundled sample's ground) get too little motion, and small fast objects are followed less closely, because its window is 15 pixels wide: on the bundled sample's ball it is 0.81 px off per field (built-in 0.23 px; a 9-pixel window 0.12 px, but then known shifts on 6_DroneFlight get worse, 0.11 against 0.07). Motion compensation shrinks the sample's pooled map to 0.64 of its size with it (the built-in estimate: under 0.6).

## 4. Pooling, motion and a learned predictor (report 3)

Salient360, the maps of section 2 that existed then, plus "pooled ±1 s + built-in motion, before / after the fix" (the built-in estimate run outside the browser on the same frames as DIS). Short names in the per-video tables: before = built-in motion before the fix, after = after the fix.

| map | kept | added, × dataset area | held-out hit rate | hits per added area, × chance | AUC-Judd |
|---|---:|---:|---:|---:|---:|
| this frame | 0.970 | 0.00 | 0.265 | — | 0.686 |
| pooled ±1 s | 0.580 | 0.51 | 0.356 | 8.59 | 0.807 |
| pooled ±1 s + motion | 0.578 | 0.55 | 0.362 | 8.45 | 0.810 |
| pooled ±1 s + built-in motion, before the fix | 0.565 | 0.54 | 0.360 | 8.39 | 0.811 |
| pooled ±1 s + built-in motion, after the fix | 0.568 | 0.53 | 0.360 | 8.48 | 0.810 |
| past 1 s | 0.531 | 0.51 | 0.321 | 7.51 | 0.774 |
| past 1 s + motion | 0.535 | 0.56 | 0.328 | 7.40 | 0.777 |
| this frame, wider Gaussian | 0.978 | 0.27 | 0.313 | 5.68 | 0.709 |
| SalFormer360-ft | 0.881 | 13.96 | 0.887 | 1.73 | 0.796 |

Test clips 1, 4, 7:

| map | kept | added, × dataset area | held-out hit rate | hits per added area, × chance | AUC-Judd |
|---|---:|---:|---:|---:|---:|
| this frame | 0.967 | 0.00 | 0.249 | — | 0.683 |
| pooled ±1 s | 0.604 | 0.45 | 0.363 | 8.95 | 0.817 |
| pooled ±1 s + motion | 0.606 | 0.47 | 0.375 | 9.00 | 0.820 |
| pooled ±1 s + built-in motion, before the fix | 0.574 | 0.48 | 0.365 | 8.64 | 0.820 |
| pooled ±1 s + built-in motion, after the fix | 0.585 | 0.48 | 0.368 | 8.71 | 0.820 |
| past 1 s | 0.552 | 0.50 | 0.319 | 7.59 | 0.782 |
| past 1 s + motion | 0.551 | 0.52 | 0.328 | 7.62 | 0.786 |
| this frame, wider Gaussian | 0.975 | 0.16 | 0.310 | 6.28 | 0.713 |
| SalFormer360-ft | 0.845 | 9.11 | 0.853 | 2.18 | 0.825 |

"this frame, wider Gaussian" was actually enlarged on 802 of 1887 frames (median σ 2.6°, salient area within 0.1 % of "pooled ±1 s", median); on the other frames "pooled ±1 s" has no larger salient area than "this frame", so the two are the same. Correlation of "this frame" with the dataset map: 0.998 (all frames), the check that OverGaze reproduces the dataset.

**Kept, per video**

| video | this frame | pooled ±1 s | + motion | before | after | past 1 s | past + motion | wider Gaussian | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.964 | 0.558 | 0.558 | 0.559 | 0.559 | 0.524 | 0.524 | 0.970 | 0.931 |
| 2_Diner | 0.972 | 0.589 | 0.589 | 0.592 | 0.591 | 0.548 | 0.547 | 0.979 | 0.959 |
| 3_PlanEnergyBioLab | 0.968 | 0.530 | 0.530 | 0.531 | 0.530 | 0.506 | 0.507 | 0.964 | 0.986 |
| 4_Ocean (test clip) | 0.972 | 0.681 | 0.684 | 0.596 | 0.627 | 0.609 | 0.606 | 0.985 | 0.826 |
| 5_Waterpark | 0.978 | 0.648 | 0.605 | 0.531 | 0.533 | 0.563 | 0.611 | 0.973 | 0.821 |
| 6_DroneFlight | 0.962 | 0.797 | 0.825 | 0.818 | 0.820 | 0.654 | 0.666 | 0.994 | 0.950 |
| 7_GazaFishermen (test clip) | 0.964 | 0.559 | 0.560 | 0.561 | 0.560 | 0.511 | 0.512 | 0.967 | 0.782 |
| 8_Sofa | 0.973 | 0.465 | 0.466 | 0.466 | 0.466 | 0.449 | 0.449 | 0.974 | 0.848 |
| 9_MattSwift | 0.975 | 0.333 | 0.332 | 0.347 | 0.340 | 0.341 | 0.342 | 0.975 | 0.678 |
| 10_Cows | 0.965 | 0.541 | 0.538 | 0.530 | 0.530 | 0.510 | 0.510 | 0.961 | 0.874 |
| 11_Abbottsford | 0.972 | 0.463 | 0.463 | 0.469 | 0.466 | 0.455 | 0.455 | 0.976 | 0.824 |
| 12_TeatroRegioTorino | 0.976 | 0.486 | 0.484 | 0.489 | 0.491 | 0.470 | 0.472 | 0.979 | 0.969 |
| 13_Fountain | 0.970 | 0.671 | 0.673 | 0.673 | 0.673 | 0.598 | 0.599 | 0.994 | 0.875 |
| 14_Warship | 0.966 | 0.512 | 0.510 | 0.506 | 0.503 | 0.478 | 0.482 | 0.969 | 0.913 |
| 15_Cockpit | 0.967 | 0.450 | 0.463 | 0.453 | 0.466 | 0.462 | 0.485 | 0.967 | 0.857 |
| 16_Turtle | 0.977 | 0.736 | 0.715 | 0.675 | 0.677 | 0.669 | 0.668 | 0.973 | 0.939 |
| 17_UnderwaterPark | 0.972 | 0.511 | 0.512 | 0.490 | 0.496 | 0.477 | 0.466 | 0.979 | 0.935 |
| 18_Bar | 0.964 | 0.682 | 0.678 | 0.676 | 0.679 | 0.604 | 0.613 | 0.986 | 0.950 |
| 19_Touvet | 0.969 | 0.779 | 0.779 | 0.772 | 0.777 | 0.636 | 0.637 | 0.998 | 0.856 |
| all 19 videos | 0.970 | 0.580 | 0.578 | 0.565 | 0.568 | 0.531 | 0.535 | 0.978 | 0.881 |
| test clips 1, 4, 7 | 0.967 | 0.604 | 0.606 | 0.574 | 0.585 | 0.552 | 0.551 | 0.975 | 0.845 |

**Added, × dataset area, per video**

| video | this frame | pooled ±1 s | + motion | before | after | past 1 s | past + motion | wider Gaussian | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.01 | 0.35 | 0.35 | 0.35 | 0.35 | 0.45 | 0.45 | 0.06 | 9.59 |
| 2_Diner | 0.00 | 0.27 | 0.27 | 0.28 | 0.27 | 0.34 | 0.33 | 0.02 | 16.11 |
| 3_PlanEnergyBioLab | 0.00 | 0.27 | 0.27 | 0.29 | 0.28 | 0.34 | 0.34 | 0.05 | 21.39 |
| 4_Ocean (test clip) | 0.00 | 0.52 | 0.59 | 0.60 | 0.60 | 0.52 | 0.57 | 0.26 | 9.82 |
| 5_Waterpark | 0.00 | 0.53 | 0.83 | 0.64 | 0.60 | 0.52 | 1.07 | 0.30 | 13.82 |
| 6_DroneFlight | 0.00 | 1.13 | 1.35 | 1.36 | 1.33 | 0.87 | 0.97 | 0.94 | 9.24 |
| 7_GazaFishermen (test clip) | 0.01 | 0.46 | 0.46 | 0.47 | 0.47 | 0.54 | 0.54 | 0.15 | 7.86 |
| 8_Sofa | 0.00 | 0.31 | 0.31 | 0.31 | 0.31 | 0.41 | 0.41 | 0.06 | 11.98 |
| 9_MattSwift | 0.00 | 0.10 | 0.10 | 0.12 | 0.11 | 0.19 | 0.19 | 0.00 | 14.56 |
| 10_Cows | 0.01 | 0.25 | 0.24 | 0.24 | 0.24 | 0.31 | 0.30 | 0.03 | 13.46 |
| 11_Abbottsford | 0.00 | 0.24 | 0.24 | 0.25 | 0.25 | 0.32 | 0.32 | 0.04 | 13.58 |
| 12_TeatroRegioTorino | 0.00 | 0.28 | 0.27 | 0.30 | 0.29 | 0.36 | 0.36 | 0.07 | 22.34 |
| 13_Fountain | 0.00 | 0.64 | 0.64 | 0.65 | 0.64 | 0.65 | 0.65 | 0.34 | 8.89 |
| 14_Warship | 0.00 | 0.29 | 0.29 | 0.30 | 0.28 | 0.36 | 0.37 | 0.04 | 22.36 |
| 15_Cockpit | 0.01 | 0.12 | 0.15 | 0.17 | 0.17 | 0.22 | 0.27 | 0.01 | 26.21 |
| 16_Turtle | 0.01 | 0.28 | 0.23 | 0.21 | 0.19 | 0.30 | 0.29 | 0.10 | 23.24 |
| 17_UnderwaterPark | 0.00 | 0.35 | 0.37 | 0.37 | 0.35 | 0.47 | 0.49 | 0.08 | 19.58 |
| 18_Bar | 0.00 | 0.53 | 0.55 | 0.55 | 0.55 | 0.52 | 0.56 | 0.25 | 16.92 |
| 19_Touvet | 0.00 | 1.24 | 1.24 | 1.23 | 1.24 | 0.98 | 0.99 | 1.03 | 8.36 |
| all 19 videos | 0.00 | 0.51 | 0.55 | 0.54 | 0.53 | 0.51 | 0.56 | 0.27 | 13.96 |
| test clips 1, 4, 7 | 0.00 | 0.45 | 0.47 | 0.48 | 0.48 | 0.50 | 0.52 | 0.16 | 9.11 |

**Held-out hit rate, per video**

| video | this frame | pooled ±1 s | + motion | before | after | past 1 s | past + motion | wider Gaussian | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.244 | 0.348 | 0.349 | 0.349 | 0.349 | 0.313 | 0.314 | 0.286 | 0.936 |
| 2_Diner | 0.278 | 0.390 | 0.390 | 0.395 | 0.393 | 0.353 | 0.354 | 0.308 | 0.964 |
| 3_PlanEnergyBioLab | 0.338 | 0.396 | 0.396 | 0.399 | 0.398 | 0.368 | 0.369 | 0.368 | 0.986 |
| 4_Ocean (test clip) | 0.286 | 0.419 | 0.450 | 0.423 | 0.431 | 0.359 | 0.383 | 0.367 | 0.842 |
| 5_Waterpark | 0.351 | 0.442 | 0.491 | 0.457 | 0.462 | 0.389 | 0.459 | 0.433 | 0.832 |
| 6_DroneFlight | 0.101 | 0.220 | 0.237 | 0.238 | 0.232 | 0.168 | 0.178 | 0.199 | 0.952 |
| 7_GazaFishermen (test clip) | 0.210 | 0.311 | 0.311 | 0.312 | 0.311 | 0.276 | 0.276 | 0.264 | 0.782 |
| 8_Sofa | 0.217 | 0.298 | 0.298 | 0.298 | 0.298 | 0.276 | 0.276 | 0.252 | 0.852 |
| 9_MattSwift | 0.257 | 0.280 | 0.280 | 0.286 | 0.284 | 0.273 | 0.273 | 0.261 | 0.680 |
| 10_Cows | 0.317 | 0.407 | 0.406 | 0.407 | 0.406 | 0.377 | 0.378 | 0.340 | 0.882 |
| 11_Abbottsford | 0.287 | 0.351 | 0.350 | 0.353 | 0.352 | 0.332 | 0.333 | 0.307 | 0.830 |
| 12_TeatroRegioTorino | 0.263 | 0.324 | 0.328 | 0.332 | 0.333 | 0.304 | 0.308 | 0.294 | 0.969 |
| 13_Fountain | 0.174 | 0.319 | 0.319 | 0.319 | 0.319 | 0.264 | 0.265 | 0.260 | 0.884 |
| 14_Warship | 0.295 | 0.392 | 0.394 | 0.394 | 0.393 | 0.357 | 0.360 | 0.325 | 0.921 |
| 15_Cockpit | 0.321 | 0.346 | 0.352 | 0.348 | 0.354 | 0.341 | 0.349 | 0.327 | 0.859 |
| 16_Turtle | 0.525 | 0.563 | 0.556 | 0.549 | 0.548 | 0.535 | 0.537 | 0.538 | 0.934 |
| 17_UnderwaterPark | 0.196 | 0.259 | 0.267 | 0.273 | 0.267 | 0.238 | 0.242 | 0.230 | 0.938 |
| 18_Bar | 0.247 | 0.417 | 0.430 | 0.429 | 0.428 | 0.356 | 0.367 | 0.340 | 0.956 |
| 19_Touvet | 0.113 | 0.263 | 0.267 | 0.270 | 0.269 | 0.203 | 0.206 | 0.233 | 0.867 |
| all 19 videos | 0.265 | 0.356 | 0.362 | 0.360 | 0.360 | 0.321 | 0.328 | 0.313 | 0.887 |
| test clips 1, 4, 7 | 0.249 | 0.363 | 0.375 | 0.365 | 0.368 | 0.319 | 0.328 | 0.310 | 0.853 |

**Hits per added area, × chance, per video**

| video | this frame | pooled ±1 s | + motion | before | after | past 1 s | past + motion | wider Gaussian | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | — | 9.71 | 9.74 | 9.71 | 9.75 | 7.96 | 7.97 | 5.73 | 2.30 |
| 2_Diner | — | 15.54 | 15.58 | 15.48 | 15.43 | 12.25 | 12.31 | 8.06 | 1.67 |
| 3_PlanEnergyBioLab | — | 12.24 | 12.24 | 12.09 | 12.19 | 9.56 | 9.57 | 10.46 | 1.54 |
| 4_Ocean (test clip) | — | 10.45 | 10.40 | 9.48 | 9.63 | 8.88 | 8.85 | 8.17 | 2.01 |
| 5_Waterpark | — | 10.59 | 9.25 | 9.77 | 10.11 | 9.10 | 7.53 | 10.49 | 1.86 |
| 6_DroneFlight | — | 3.45 | 3.27 | 3.23 | 3.24 | 3.24 | 3.21 | 2.97 | 1.72 |
| 7_GazaFishermen (test clip) | — | 6.94 | 6.93 | 6.92 | 6.92 | 6.07 | 6.06 | 4.70 | 2.26 |
| 8_Sofa | — | 7.94 | 7.94 | 7.92 | 7.95 | 6.64 | 6.63 | 5.10 | 1.84 |
| 9_MattSwift | — | 11.81 | 12.09 | 11.77 | 11.93 | 7.62 | 7.63 | 6.81 | 1.85 |
| 10_Cows | — | 16.11 | 16.14 | 16.20 | 16.21 | 12.41 | 12.46 | 8.99 | 2.01 |
| 11_Abbottsford | — | 12.32 | 12.33 | 12.36 | 12.37 | 9.96 | 9.98 | 6.28 | 1.94 |
| 12_TeatroRegioTorino | — | 10.13 | 10.54 | 10.26 | 10.54 | 8.10 | 8.33 | 7.50 | 1.48 |
| 13_Fountain | — | 7.01 | 6.96 | 6.96 | 6.96 | 6.28 | 6.28 | 4.90 | 2.01 |
| 14_Warship | — | 12.46 | 12.54 | 12.29 | 12.43 | 9.49 | 9.62 | 7.67 | 1.25 |
| 15_Cockpit | — | 15.38 | 14.37 | 13.45 | 13.89 | 10.55 | 9.99 | 7.48 | 1.50 |
| 16_Turtle | — | 29.02 | 30.51 | 29.67 | 31.51 | 22.55 | 22.79 | 17.81 | 3.23 |
| 17_UnderwaterPark | — | 6.08 | 6.19 | 6.48 | 6.47 | 5.16 | 5.30 | 4.48 | 1.21 |
| 18_Bar | — | 10.90 | 10.88 | 10.74 | 10.95 | 9.62 | 9.40 | 7.58 | 1.33 |
| 19_Touvet | — | 3.59 | 3.66 | 3.70 | 3.69 | 3.49 | 3.54 | 3.04 | 1.76 |
| all 19 videos | — | 8.59 | 8.45 | 8.39 | 8.48 | 7.51 | 7.40 | 5.68 | 1.73 |
| test clips 1, 4, 7 | — | 8.95 | 9.00 | 8.64 | 8.71 | 7.59 | 7.62 | 6.28 | 2.18 |

**AUC-Judd, per video**

| video | this frame | pooled ±1 s | + motion | before | after | past 1 s | past + motion | wider Gaussian | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.680 | 0.840 | 0.841 | 0.841 | 0.841 | 0.799 | 0.799 | 0.705 | 0.865 |
| 2_Diner | 0.700 | 0.841 | 0.841 | 0.842 | 0.841 | 0.802 | 0.803 | 0.714 | 0.792 |
| 3_PlanEnergyBioLab | 0.734 | 0.844 | 0.844 | 0.845 | 0.844 | 0.814 | 0.814 | 0.746 | 0.875 |
| 4_Ocean (test clip) | 0.712 | 0.819 | 0.827 | 0.826 | 0.826 | 0.792 | 0.802 | 0.750 | 0.816 |
| 5_Waterpark | 0.749 | 0.836 | 0.853 | 0.851 | 0.853 | 0.810 | 0.829 | 0.785 | 0.773 |
| 6_DroneFlight | 0.578 | 0.721 | 0.728 | 0.730 | 0.729 | 0.670 | 0.678 | 0.626 | 0.784 |
| 7_GazaFishermen (test clip) | 0.651 | 0.791 | 0.792 | 0.792 | 0.792 | 0.753 | 0.753 | 0.678 | 0.794 |
| 8_Sofa | 0.647 | 0.767 | 0.767 | 0.768 | 0.767 | 0.734 | 0.734 | 0.663 | 0.791 |
| 9_MattSwift | 0.672 | 0.767 | 0.767 | 0.767 | 0.767 | 0.740 | 0.740 | 0.673 | 0.736 |
| 10_Cows | 0.717 | 0.836 | 0.836 | 0.836 | 0.837 | 0.805 | 0.805 | 0.727 | 0.854 |
| 11_Abbottsford | 0.698 | 0.831 | 0.831 | 0.831 | 0.831 | 0.795 | 0.795 | 0.709 | 0.802 |
| 12_TeatroRegioTorino | 0.685 | 0.808 | 0.810 | 0.812 | 0.812 | 0.774 | 0.777 | 0.701 | 0.809 |
| 13_Fountain | 0.637 | 0.795 | 0.796 | 0.797 | 0.796 | 0.755 | 0.757 | 0.678 | 0.834 |
| 14_Warship | 0.698 | 0.789 | 0.789 | 0.790 | 0.789 | 0.766 | 0.767 | 0.710 | 0.751 |
| 15_Cockpit | 0.715 | 0.820 | 0.824 | 0.826 | 0.826 | 0.792 | 0.796 | 0.718 | 0.703 |
| 16_Turtle | 0.825 | 0.919 | 0.921 | 0.923 | 0.922 | 0.893 | 0.895 | 0.833 | 0.867 |
| 17_UnderwaterPark | 0.639 | 0.743 | 0.747 | 0.749 | 0.746 | 0.712 | 0.715 | 0.657 | 0.753 |
| 18_Bar | 0.694 | 0.832 | 0.837 | 0.837 | 0.837 | 0.799 | 0.805 | 0.740 | 0.764 |
| 19_Touvet | 0.585 | 0.733 | 0.735 | 0.736 | 0.735 | 0.690 | 0.692 | 0.649 | 0.767 |
| all 19 videos | 0.686 | 0.807 | 0.810 | 0.811 | 0.810 | 0.774 | 0.777 | 0.709 | 0.796 |
| test clips 1, 4, 7 | 0.683 | 0.817 | 0.820 | 0.820 | 0.820 | 0.782 | 0.786 | 0.713 | 0.825 |

**Example frames (numbers only; maps from one half of the people, split 1)**

*14_Warship, frame 460: "pooled ±1 s" adds area that catches the other half's fixations; "this frame, wider Gaussian", adding about as much, does not.*

| map | kept | added, × dataset area | held-out hit rate | other half's fixations in the added area | added area, % of sphere | split 1: fixations in the added area |
|---|---:|---:|---:|---:|---:|---:|
| this frame | 86.0 % | 0.00 | 12.4 % | 0.0 % | 0.00 | 0 of 37 |
| pooled ±1 s | 53.3 % | 0.31 | 38.3 % | 25.9 % | 1.87 | 8 of 37 |
| pooled ±1 s + motion | 51.9 % | 0.29 | 37.7 % | 25.3 % | 1.85 | 7 of 37 |
| pooled ±1 s + built-in motion, before the fix | 49.6 % | 0.24 | 37.1 % | 24.7 % | 1.77 | 7 of 37 |
| pooled ±1 s + built-in motion, after the fix | 50.6 % | 0.25 | 37.4 % | 25.0 % | 1.80 | 7 of 37 |
| past 1 s | 42.7 % | 0.23 | 29.9 % | 17.7 % | 1.46 | 3 of 37 |
| past 1 s + motion | 42.5 % | 0.22 | 29.9 % | 17.7 % | 1.45 | 3 of 37 |
| this frame, wider Gaussian | 86.0 % | 0.00 | 16.0 % | 3.6 % | 1.07 | 0 of 37 |
| SalFormer360-ft | 94.7 % | 17.71 | 94.4 % | 81.9 % | 65.85 | 29 of 37 |

*17_UnderwaterPark, frame 495: "pooled ±1 s" does not beat "this frame, wider Gaussian".*

| map | kept | added, × dataset area | held-out hit rate | other half's fixations in the added area | added area, % of sphere | split 1: fixations in the added area |
|---|---:|---:|---:|---:|---:|---:|
| this frame | 97.0 % | 0.00 | 14.6 % | 0.0 % | 0.00 | 0 of 48 |
| pooled ±1 s | 57.5 % | 0.35 | 17.3 % | 5.2 % | 2.66 | 2 of 48 |
| pooled ±1 s + motion | 57.7 % | 0.41 | 18.5 % | 6.2 % | 2.80 | 2 of 48 |
| pooled ±1 s + built-in motion, before the fix | 55.5 % | 0.43 | 19.0 % | 6.5 % | 2.78 | 1 of 48 |
| pooled ±1 s + built-in motion, after the fix | 56.5 % | 0.37 | 21.1 % | 8.1 % | 2.73 | 1 of 48 |
| past 1 s | 43.2 % | 0.38 | 14.8 % | 4.8 % | 2.36 | 2 of 48 |
| past 1 s + motion | 40.5 % | 0.39 | 17.3 % | 6.9 % | 2.48 | 1 of 48 |
| this frame, wider Gaussian | 97.0 % | 0.00 | 20.2 % | 5.6 % | 1.81 | 1 of 48 |
| SalFormer360-ft | 84.9 % | 10.81 | 86.4 % | 71.8 % | 58.94 | 34 of 48 |

*16_Turtle, frame 440: the frame where motion compensation failed in the first figure.*

| map | kept | added, × dataset area | held-out hit rate | other half's fixations in the added area | added area, % of sphere | split 1: fixations in the added area |
|---|---:|---:|---:|---:|---:|---:|
| this frame | 97.6 % | 0.00 | 79.8 % | 0.0 % | 0.00 | 0 of 53 |
| pooled ±1 s | 90.4 % | 0.05 | 79.3 % | 0.9 % | 0.02 | 0 of 53 |
| pooled ±1 s + motion | 95.2 % | 0.30 | 80.6 % | 1.7 % | 0.13 | 1 of 53 |
| pooled ±1 s + built-in motion, before the fix | 65.2 % | 0.81 | 71.7 % | 3.1 % | 0.33 | 2 of 53 |
| pooled ±1 s + built-in motion, after the fix | 67.6 % | 0.33 | 73.4 % | 1.1 % | 0.14 | 0 of 53 |
| past 1 s | 86.6 % | 0.10 | 78.5 % | 0.6 % | 0.05 | 1 of 53 |
| past 1 s + motion | 96.3 % | 1.20 | 81.7 % | 2.4 % | 0.57 | 2 of 53 |
| this frame, wider Gaussian | 97.6 % | 0.00 | 80.2 % | 0.4 % | 0.01 | 0 of 53 |
| SalFormer360-ft | 82.1 % | 76.06 | 90.5 % | 14.0 % | 29.95 | 5 of 53 |

**Conclusions.**

- Pooling over ±1 s predicts other people better than the current frame alone on all 19 videos (held-out hit rate 0.356 against 0.265, AUC-Judd 0.807 against 0.686), although it keeps only 58 % of the dataset's own map: that map is built from the same people at the same moment, so keeping it is not the goal.
- A wider Gaussian on the current frame is not a substitute: it loses to pooling on all 19 videos on hit rate, hits per added area and AUC-Judd. What pooling adds is where people really look a moment earlier or later.
- Motion compensation with OpenCV motion helps a little (hit rate higher on 13 of 19 videos, AUC-Judd on 18 of 19). The built-in estimate before and after the fix gives about the same averages, but fails on single frames such as 16_Turtle frame 440.
- Past-only pooling (a live mask) is below centred pooling on all 19 videos; adding motion improves it on 18 of 19.
- SalFormer360-ft ranks places about as well as pooled gaze on AUC-Judd (0.796 against 0.807; better on the test clips, 0.825 against 0.817), but its salient area covers about half of the sphere, so per unit of added area it is only 1.7 × chance against 8.6 × for pooling.

## 5. Weighting the current frame and scene memory (report 4)

All 19 videos:

| map | kept | added, × dataset area | held-out hit rate | hits per added area, × chance | AUC-Judd | shuffled AUC, within video | NSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| this frame | 0.970 | 0.00 | 0.265 | — | 0.686 | 0.564 | 2.57 |
| this frame, wider Gaussian | 0.978 | 0.27 | 0.313 | 5.68 | 0.709 | 0.567 | 2.66 |
| pooled ±1 s | 0.580 | 0.51 | 0.356 | 8.59 | 0.807 | 0.574 | 3.29 |
| pooled ±1 s + motion | 0.578 | 0.55 | 0.362 | 8.45 | 0.810 | 0.575 | 3.30 |
| past 1 s | 0.531 | 0.51 | 0.321 | 7.51 | 0.774 | 0.571 | 2.99 |
| past 1 s + motion | 0.535 | 0.56 | 0.328 | 7.40 | 0.777 | 0.572 | 2.98 |
| this frame + pooled | 0.812 | 0.21 | 0.322 | 11.90 | 0.806 | 0.574 | 3.07 |
| pooled, peaked τ = 0.1 s | 0.813 | 0.16 | 0.294 | 9.40 | 0.803 | 0.575 | 2.88 |
| pooled, peaked τ = 0.1 s + motion | 0.811 | 0.17 | 0.298 | 9.36 | 0.806 | 0.575 | 2.90 |
| pooled, peaked τ = 0.25 s | 0.687 | 0.32 | 0.329 | 9.39 | 0.806 | 0.575 | 3.15 |
| pooled, peaked τ = 0.25 s + motion | 0.686 | 0.36 | 0.336 | 9.27 | 0.809 | 0.576 | 3.17 |
| pooled, peaked τ = 0.5 s | 0.610 | 0.46 | 0.352 | 8.97 | 0.807 | 0.574 | 3.29 |
| pooled, peaked τ = 0.5 s + motion | 0.609 | 0.50 | 0.360 | 8.82 | 0.810 | 0.576 | 3.30 |
| past 1 s, peaked τ = 0.1 s | 0.801 | 0.17 | 0.285 | 8.29 | 0.772 | 0.573 | 2.77 |
| past 1 s, peaked τ = 0.1 s + motion | 0.798 | 0.18 | 0.287 | 8.23 | 0.775 | 0.573 | 2.77 |
| past 1 s, peaked τ = 0.25 s | 0.657 | 0.34 | 0.307 | 8.12 | 0.773 | 0.572 | 2.94 |
| past 1 s, peaked τ = 0.25 s + motion | 0.659 | 0.37 | 0.313 | 8.05 | 0.777 | 0.573 | 2.95 |
| past 1 s, peaked τ = 0.5 s | 0.569 | 0.47 | 0.322 | 7.79 | 0.774 | 0.571 | 3.01 |
| past 1 s, peaked τ = 0.5 s + motion | 0.574 | 0.51 | 0.330 | 7.67 | 0.777 | 0.572 | 3.01 |
| pooled + memory w = 0.1 | 0.687 | 0.39 | 0.352 | 9.89 | 0.843 | 0.566 | 3.25 |
| pooled + memory w = 0.3 | 0.678 | 0.50 | 0.382 | 10.03 | 0.846 | 0.561 | 3.30 |
| past 1 s + memory w = 0.1 | 0.662 | 0.40 | 0.328 | 8.69 | 0.835 | 0.561 | 3.03 |
| past 1 s + memory w = 0.3 | 0.657 | 0.50 | 0.359 | 9.25 | 0.838 | 0.556 | 3.12 |
| SalFormer360-ft | 0.881 | 13.96 | 0.887 | 1.73 | 0.796 | 0.526 | 1.25 |

Test clips 1, 4, 7:

| map | kept | added, × dataset area | held-out hit rate | hits per added area, × chance | AUC-Judd | shuffled AUC, within video | NSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| this frame | 0.967 | 0.00 | 0.249 | — | 0.683 | 0.565 | 1.89 |
| this frame, wider Gaussian | 0.975 | 0.16 | 0.310 | 6.28 | 0.713 | 0.568 | 2.02 |
| pooled ±1 s | 0.604 | 0.45 | 0.363 | 8.95 | 0.817 | 0.577 | 2.59 |
| pooled ±1 s + motion | 0.606 | 0.47 | 0.375 | 9.00 | 0.820 | 0.579 | 2.62 |
| past 1 s | 0.552 | 0.50 | 0.319 | 7.59 | 0.782 | 0.573 | 2.29 |
| past 1 s + motion | 0.551 | 0.52 | 0.328 | 7.62 | 0.786 | 0.574 | 2.31 |
| this frame + pooled | 0.853 | 0.17 | 0.320 | 12.65 | 0.816 | 0.578 | 2.38 |
| pooled, peaked τ = 0.1 s | 0.844 | 0.15 | 0.286 | 9.22 | 0.812 | 0.578 | 2.19 |
| pooled, peaked τ = 0.1 s + motion | 0.845 | 0.17 | 0.290 | 9.37 | 0.815 | 0.579 | 2.20 |
| pooled, peaked τ = 0.25 s | 0.727 | 0.30 | 0.330 | 9.55 | 0.815 | 0.578 | 2.45 |
| pooled, peaked τ = 0.25 s + motion | 0.730 | 0.33 | 0.340 | 9.65 | 0.818 | 0.580 | 2.47 |
| pooled, peaked τ = 0.5 s | 0.640 | 0.40 | 0.359 | 9.36 | 0.817 | 0.577 | 2.58 |
| pooled, peaked τ = 0.5 s + motion | 0.643 | 0.43 | 0.371 | 9.43 | 0.820 | 0.580 | 2.61 |
| past 1 s, peaked τ = 0.1 s | 0.823 | 0.17 | 0.273 | 8.28 | 0.780 | 0.575 | 2.08 |
| past 1 s, peaked τ = 0.1 s + motion | 0.818 | 0.18 | 0.275 | 8.23 | 0.784 | 0.576 | 2.09 |
| past 1 s, peaked τ = 0.25 s | 0.687 | 0.34 | 0.302 | 8.22 | 0.782 | 0.575 | 2.24 |
| past 1 s, peaked τ = 0.25 s + motion | 0.683 | 0.36 | 0.308 | 8.24 | 0.786 | 0.576 | 2.26 |
| past 1 s, peaked τ = 0.5 s | 0.595 | 0.46 | 0.321 | 7.88 | 0.782 | 0.573 | 2.31 |
| past 1 s, peaked τ = 0.5 s + motion | 0.595 | 0.48 | 0.330 | 7.91 | 0.786 | 0.575 | 2.33 |
| pooled + memory w = 0.1 | 0.722 | 0.35 | 0.358 | 10.34 | 0.848 | 0.566 | 2.56 |
| pooled + memory w = 0.3 | 0.702 | 0.40 | 0.384 | 10.37 | 0.851 | 0.559 | 2.67 |
| past 1 s + memory w = 0.1 | 0.681 | 0.38 | 0.325 | 8.96 | 0.840 | 0.558 | 2.36 |
| past 1 s + memory w = 0.3 | 0.669 | 0.43 | 0.355 | 9.43 | 0.842 | 0.552 | 2.49 |
| SalFormer360-ft | 0.845 | 9.11 | 0.853 | 2.18 | 0.825 | 0.541 | 1.46 |

**Does "kept" get back to about 0.97 while the held-out scores stay at or above "pooled ±1 s"?** No:

|  | kept | held-out hit rate | AUC-Judd | NSS |
|---|---:|---:|---:|---:|
| pooled ±1 s | 0.58 | 0.356 | 0.807 | 3.29 |
| pooled, peaked τ = 0.1 s + motion | 0.81 | 0.298 | 0.806 | 2.90 |
| pooled, peaked τ = 0.25 s + motion | 0.69 | 0.336 | 0.809 | 3.17 |
| pooled, peaked τ = 0.5 s + motion | 0.61 | 0.360 | 0.810 | 3.30 |

**Scene memory tests.**

One half's map of the first 3 s against the other half's fixations of the first 3 s (mean of 19 videos × 10 half comparisons, no motion): hit rate 41.7 % in 3.2 % of the sphere, AUC-Judd 0.849, shuffled AUC within video 0.668 (comparison fixations after 5 s), NSS 2.95.

Looked at again later: the salient area of everyone's first 3 s, moved along the motion to a later frame, and the share of that frame's fixations inside it; next to it a region with the same habits, made the same way from a recent 3 s window (from 6 s to 3 s before the frame).

| later frames | fixations inside the first-3-s region | its area | × its area | fixations inside the recent region | its area |
|---|---:|---:|---:|---:|---:|
| 3–6 s | 31.0 % | 3.1 % | 9.9 |  |  |
| 6–10 s | 25.8 % | 3.0 % | 8.7 | 31.3 % | 3.7 % |
| 10–20 s | 21.3 % | 2.6 % | 8.3 | 32.8 % | 4.6 % |

**Significance against "pooled ±1 s"** (per-video intervals: block bootstrap; counts over 19 videos):

*AUC-Judd*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| this frame | -0.122 | 0 / 19 | 0 / 19 | < 0.001 |
| this frame, wider Gaussian | -0.099 | 0 / 19 | 0 / 19 | < 0.001 |
| pooled ±1 s + motion | 0.003 | 18 / 1 | 13 / 0 | < 0.001 |
| past 1 s | -0.034 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s + motion | -0.030 | 0 / 19 | 0 / 19 | < 0.001 |
| this frame + pooled | -0.001 | 1 / 18 | 1 / 16 | < 0.001 |
| pooled, peaked τ = 0.1 s | -0.004 | 1 / 18 | 1 / 18 | < 0.001 |
| pooled, peaked τ = 0.1 s + motion | -0.002 | 3 / 16 | 3 / 11 | 0.004 |
| pooled, peaked τ = 0.25 s | -0.001 | 1 / 18 | 1 / 18 | < 0.001 |
| pooled, peaked τ = 0.25 s + motion | 0.001 | 9 / 10 | 6 / 8 | 1.000 |
| pooled, peaked τ = 0.5 s | 0.000 | 16 / 3 | 7 / 0 | 0.004 |
| pooled, peaked τ = 0.5 s + motion | 0.003 | 18 / 1 | 16 / 0 | < 0.001 |
| past 1 s, peaked τ = 0.1 s | -0.035 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.1 s + motion | -0.032 | 0 / 19 | 0 / 18 | < 0.001 |
| past 1 s, peaked τ = 0.25 s | -0.034 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.25 s + motion | -0.031 | 0 / 19 | 0 / 18 | < 0.001 |
| past 1 s, peaked τ = 0.5 s | -0.033 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.5 s + motion | -0.030 | 0 / 19 | 0 / 19 | < 0.001 |
| pooled + memory w = 0.1 | 0.036 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled + memory w = 0.3 | 0.039 | 19 / 0 | 19 / 0 | < 0.001 |
| past 1 s + memory w = 0.1 | 0.028 | 19 / 0 | 19 / 0 | < 0.001 |
| past 1 s + memory w = 0.3 | 0.031 | 19 / 0 | 19 / 0 | < 0.001 |
| SalFormer360-ft | -0.011 | 10 / 9 | 8 / 8 | 1.000 |

*shuffled AUC, within video*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| this frame | -0.010 | 2 / 17 | 1 / 11 | < 0.001 |
| this frame, wider Gaussian | -0.007 | 3 / 16 | 1 / 7 | 0.004 |
| pooled ±1 s + motion | 0.002 | 17 / 2 | 10 / 0 | < 0.001 |
| past 1 s | -0.003 | 6 / 13 | 0 / 6 | 0.167 |
| past 1 s + motion | -0.002 | 7 / 12 | 0 / 4 | 0.359 |
| this frame + pooled | 0.001 | 12 / 7 | 4 / 4 | 0.359 |
| pooled, peaked τ = 0.1 s | 0.001 | 10 / 9 | 5 / 4 | 1.000 |
| pooled, peaked τ = 0.1 s + motion | 0.002 | 11 / 8 | 5 / 3 | 0.648 |
| pooled, peaked τ = 0.25 s | 0.001 | 13 / 6 | 5 / 2 | 0.167 |
| pooled, peaked τ = 0.25 s + motion | 0.002 | 16 / 3 | 8 / 0 | 0.004 |
| pooled, peaked τ = 0.5 s | 0.000 | 14 / 5 | 5 / 1 | 0.064 |
| pooled, peaked τ = 0.5 s + motion | 0.002 | 18 / 1 | 10 / 0 | < 0.001 |
| past 1 s, peaked τ = 0.1 s | -0.001 | 9 / 10 | 2 / 6 | 1.000 |
| past 1 s, peaked τ = 0.1 s + motion | -0.001 | 10 / 9 | 2 / 5 | 1.000 |
| past 1 s, peaked τ = 0.25 s | -0.001 | 9 / 10 | 1 / 6 | 1.000 |
| past 1 s, peaked τ = 0.25 s + motion | -0.001 | 9 / 10 | 1 / 3 | 1.000 |
| past 1 s, peaked τ = 0.5 s | -0.002 | 6 / 13 | 0 / 6 | 0.167 |
| past 1 s, peaked τ = 0.5 s + motion | -0.002 | 7 / 12 | 0 / 3 | 0.359 |
| pooled + memory w = 0.1 | -0.008 | 1 / 18 | 1 / 13 | < 0.001 |
| pooled + memory w = 0.3 | -0.014 | 1 / 18 | 1 / 14 | < 0.001 |
| past 1 s + memory w = 0.1 | -0.013 | 1 / 18 | 0 / 15 | < 0.001 |
| past 1 s + memory w = 0.3 | -0.018 | 1 / 18 | 0 / 15 | < 0.001 |
| SalFormer360-ft | -0.047 | 0 / 19 | 0 / 13 | < 0.001 |

*NSS*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| this frame | -0.73 | 0 / 19 | 0 / 19 | < 0.001 |
| this frame, wider Gaussian | -0.64 | 0 / 19 | 0 / 18 | < 0.001 |
| pooled ±1 s + motion | 0.01 | 15 / 4 | 7 / 1 | 0.019 |
| past 1 s | -0.30 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s + motion | -0.31 | 0 / 19 | 0 / 19 | < 0.001 |
| this frame + pooled | -0.23 | 0 / 19 | 0 / 17 | < 0.001 |
| pooled, peaked τ = 0.1 s | -0.41 | 0 / 19 | 0 / 18 | < 0.001 |
| pooled, peaked τ = 0.1 s + motion | -0.40 | 1 / 18 | 0 / 17 | < 0.001 |
| pooled, peaked τ = 0.25 s | -0.14 | 2 / 17 | 1 / 16 | < 0.001 |
| pooled, peaked τ = 0.25 s + motion | -0.13 | 2 / 17 | 2 / 15 | < 0.001 |
| pooled, peaked τ = 0.5 s | -0.01 | 3 / 16 | 2 / 14 | 0.004 |
| pooled, peaked τ = 0.5 s + motion | 0.00 | 7 / 12 | 2 / 6 | 0.359 |
| past 1 s, peaked τ = 0.1 s | -0.53 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.1 s + motion | -0.53 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.25 s | -0.35 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.25 s + motion | -0.35 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.5 s | -0.28 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.5 s + motion | -0.28 | 0 / 19 | 0 / 19 | < 0.001 |
| pooled + memory w = 0.1 | -0.05 | 3 / 16 | 1 / 12 | 0.004 |
| pooled + memory w = 0.3 | 0.01 | 15 / 4 | 9 / 2 | 0.019 |
| past 1 s + memory w = 0.1 | -0.26 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s + memory w = 0.3 | -0.18 | 2 / 17 | 1 / 15 | < 0.001 |
| SalFormer360-ft | -2.02 | 2 / 17 | 2 / 17 | < 0.001 |

*held-out hit rate*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| this frame | -0.091 | 0 / 19 | 0 / 19 | < 0.001 |
| this frame, wider Gaussian | -0.043 | 0 / 19 | 0 / 18 | < 0.001 |
| pooled ±1 s + motion | 0.007 | 13 / 6 | 9 / 1 | 0.167 |
| past 1 s | -0.035 | 0 / 19 | 0 / 18 | < 0.001 |
| past 1 s + motion | -0.027 | 2 / 17 | 0 / 17 | < 0.001 |
| this frame + pooled | -0.034 | 0 / 19 | 0 / 19 | < 0.001 |
| pooled, peaked τ = 0.1 s | -0.061 | 0 / 19 | 0 / 19 | < 0.001 |
| pooled, peaked τ = 0.1 s + motion | -0.057 | 0 / 19 | 0 / 18 | < 0.001 |
| pooled, peaked τ = 0.25 s | -0.026 | 0 / 19 | 0 / 18 | < 0.001 |
| pooled, peaked τ = 0.25 s + motion | -0.020 | 1 / 18 | 1 / 15 | < 0.001 |
| pooled, peaked τ = 0.5 s | -0.003 | 1 / 18 | 0 / 15 | < 0.001 |
| pooled, peaked τ = 0.5 s + motion | 0.004 | 9 / 10 | 5 / 8 | 1.000 |
| past 1 s, peaked τ = 0.1 s | -0.071 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.1 s + motion | -0.069 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.25 s | -0.048 | 0 / 19 | 0 / 19 | < 0.001 |
| past 1 s, peaked τ = 0.25 s + motion | -0.043 | 1 / 18 | 0 / 17 | < 0.001 |
| past 1 s, peaked τ = 0.5 s | -0.033 | 0 / 19 | 0 / 18 | < 0.001 |
| past 1 s, peaked τ = 0.5 s + motion | -0.026 | 2 / 17 | 1 / 17 | < 0.001 |
| pooled + memory w = 0.1 | -0.004 | 7 / 12 | 3 / 9 | 0.359 |
| pooled + memory w = 0.3 | 0.026 | 18 / 1 | 15 / 0 | < 0.001 |
| past 1 s + memory w = 0.1 | -0.028 | 3 / 16 | 1 / 16 | 0.004 |
| past 1 s + memory w = 0.3 | 0.004 | 10 / 9 | 8 / 4 | 1.000 |
| SalFormer360-ft | 0.533 | 19 / 0 | 19 / 0 | < 0.001 |

*kept*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| this frame | 0.391 | 19 / 0 | 19 / 0 | < 0.001 |
| this frame, wider Gaussian | 0.399 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled ±1 s + motion | -0.002 | 9 / 10 | 2 / 1 | 1.000 |
| past 1 s | -0.049 | 2 / 17 | 0 / 14 | < 0.001 |
| past 1 s + motion | -0.044 | 2 / 17 | 1 / 13 | < 0.001 |
| this frame + pooled | 0.233 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled, peaked τ = 0.1 s | 0.234 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled, peaked τ = 0.1 s + motion | 0.233 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled, peaked τ = 0.25 s | 0.108 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled, peaked τ = 0.25 s + motion | 0.107 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled, peaked τ = 0.5 s | 0.030 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled, peaked τ = 0.5 s + motion | 0.029 | 17 / 2 | 17 / 1 | < 0.001 |
| past 1 s, peaked τ = 0.1 s | 0.222 | 19 / 0 | 19 / 0 | < 0.001 |
| past 1 s, peaked τ = 0.1 s + motion | 0.220 | 19 / 0 | 19 / 0 | < 0.001 |
| past 1 s, peaked τ = 0.25 s | 0.078 | 17 / 2 | 16 / 0 | < 0.001 |
| past 1 s, peaked τ = 0.25 s + motion | 0.080 | 16 / 3 | 16 / 0 | 0.004 |
| past 1 s, peaked τ = 0.5 s | -0.011 | 11 / 8 | 2 / 7 | 0.648 |
| past 1 s, peaked τ = 0.5 s + motion | -0.006 | 11 / 8 | 2 / 6 | 0.648 |
| pooled + memory w = 0.1 | 0.108 | 19 / 0 | 19 / 0 | < 0.001 |
| pooled + memory w = 0.3 | 0.099 | 19 / 0 | 19 / 0 | < 0.001 |
| past 1 s + memory w = 0.1 | 0.083 | 19 / 0 | 16 / 0 | < 0.001 |
| past 1 s + memory w = 0.3 | 0.078 | 19 / 0 | 18 / 0 | < 0.001 |
| SalFormer360-ft | 0.304 | 19 / 0 | 19 / 0 | < 0.001 |

*Per video, AUC-Judd difference from "pooled ±1 s" [95 % interval]*

| video | pooled ±1 s + motion | this frame + pooled | pooled, peaked τ = 0.1 s + motion | pooled, peaked τ = 0.25 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.3 | past 1 s + memory w = 0.3 | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.000 [0.000, 0.001] | -0.001 [-0.002, -0.001] | -0.006 [-0.007, -0.005] | -0.002 [-0.002, -0.001] | 0.000 [0.000, 0.001] | 0.037 [0.029, 0.043] | 0.029 [0.017, 0.038] | 0.025 [0.019, 0.031] |
| 2_Diner | 0.000 [-0.000, 0.000] | -0.002 [-0.002, -0.002] | -0.008 [-0.009, -0.007] | -0.003 [-0.003, -0.002] | 0.000 [-0.000, 0.000] | 0.043 [0.034, 0.051] | 0.034 [0.019, 0.046] | -0.049 [-0.061, -0.038] |
| 3_PlanEnergyBioLab | 0.000 [-0.000, 0.000] | -0.000 [-0.001, 0.000] | -0.002 [-0.003, -0.001] | -0.001 [-0.001, -0.000] | 0.000 [0.000, 0.000] | 0.039 [0.030, 0.048] | 0.034 [0.021, 0.044] | 0.030 [0.020, 0.041] |
| 4_Ocean (test clip) | 0.008 [0.006, 0.011] | -0.001 [-0.001, -0.000] | 0.003 [0.001, 0.006] | 0.007 [0.004, 0.009] | 0.008 [0.006, 0.011] | 0.021 [0.015, 0.026] | 0.012 [0.004, 0.018] | -0.003 [-0.014, 0.008] |
| 5_Waterpark | 0.017 [0.012, 0.021] | 0.001 [0.000, 0.001] | 0.019 [0.014, 0.023] | 0.019 [0.014, 0.023] | 0.018 [0.013, 0.022] | 0.028 [0.023, 0.032] | 0.013 [0.007, 0.019] | -0.063 [-0.082, -0.044] |
| 6_DroneFlight | 0.007 [0.003, 0.012] | -0.000 [-0.001, 0.000] | 0.004 [0.000, 0.009] | 0.006 [0.002, 0.011] | 0.008 [0.004, 0.012] | 0.064 [0.051, 0.075] | 0.053 [0.034, 0.071] | 0.063 [0.045, 0.080] |
| 7_GazaFishermen (test clip) | 0.000 [-0.000, 0.000] | -0.000 [-0.001, -0.000] | -0.004 [-0.005, -0.003] | -0.001 [-0.002, -0.001] | 0.000 [0.000, 0.000] | 0.046 [0.036, 0.055] | 0.038 [0.023, 0.051] | 0.003 [-0.024, 0.027] |
| 8_Sofa | 0.000 [0.000, 0.000] | -0.001 [-0.002, -0.001] | -0.007 [-0.008, -0.006] | -0.002 [-0.003, -0.002] | 0.000 [0.000, 0.000] | 0.046 [0.035, 0.056] | 0.040 [0.025, 0.052] | 0.024 [0.008, 0.038] |
| 9_MattSwift | 0.000 [-0.000, 0.000] | -0.001 [-0.001, -0.000] | -0.003 [-0.004, -0.002] | -0.001 [-0.001, -0.000] | 0.000 [-0.000, 0.000] | 0.036 [0.029, 0.043] | 0.029 [0.018, 0.040] | -0.031 [-0.056, -0.009] |
| 10_Cows | 0.000 [-0.000, 0.001] | -0.001 [-0.002, -0.001] | -0.005 [-0.006, -0.004] | -0.002 [-0.002, -0.001] | 0.000 [0.000, 0.001] | 0.034 [0.027, 0.041] | 0.027 [0.016, 0.037] | 0.018 [0.010, 0.026] |
| 11_Abbottsford | -0.000 [-0.000, 0.000] | -0.002 [-0.002, -0.001] | -0.007 [-0.008, -0.006] | -0.002 [-0.003, -0.002] | -0.000 [-0.000, 0.000] | 0.044 [0.035, 0.053] | 0.036 [0.023, 0.048] | -0.029 [-0.039, -0.019] |
| 12_TeatroRegioTorino | 0.002 [0.001, 0.003] | -0.001 [-0.001, -0.001] | -0.002 [-0.003, -0.001] | 0.000 [-0.000, 0.001] | 0.002 [0.001, 0.003] | 0.041 [0.035, 0.046] | 0.034 [0.024, 0.042] | 0.000 [-0.010, 0.011] |
| 13_Fountain | 0.002 [0.001, 0.002] | -0.001 [-0.001, -0.001] | -0.005 [-0.007, -0.004] | -0.001 [-0.002, 0.000] | 0.002 [0.001, 0.002] | 0.049 [0.039, 0.058] | 0.042 [0.026, 0.056] | 0.039 [0.027, 0.049] |
| 14_Warship | 0.001 [0.000, 0.001] | -0.001 [-0.002, -0.001] | -0.003 [-0.005, -0.002] | -0.001 [-0.001, 0.000] | 0.001 [0.000, 0.001] | 0.031 [0.026, 0.037] | 0.025 [0.016, 0.033] | -0.038 [-0.059, -0.020] |
| 15_Cockpit | 0.004 [0.003, 0.005] | -0.001 [-0.002, -0.001] | -0.001 [-0.003, 0.002] | 0.002 [0.001, 0.004] | 0.004 [0.003, 0.006] | 0.038 [0.031, 0.045] | 0.031 [0.020, 0.040] | -0.117 [-0.134, -0.099] |
| 16_Turtle | 0.002 [0.001, 0.003] | -0.001 [-0.001, -0.000] | -0.000 [-0.001, 0.001] | 0.001 [0.001, 0.002] | 0.002 [0.001, 0.003] | 0.020 [0.017, 0.023] | 0.013 [0.007, 0.018] | -0.052 [-0.079, -0.026] |
| 17_UnderwaterPark | 0.003 [0.001, 0.006] | -0.001 [-0.001, -0.000] | -0.001 [-0.004, 0.001] | 0.002 [-0.000, 0.004] | 0.003 [0.001, 0.006] | 0.031 [0.023, 0.038] | 0.023 [0.014, 0.032] | 0.010 [0.000, 0.019] |
| 18_Bar | 0.004 [0.003, 0.006] | -0.001 [-0.002, -0.001] | -0.002 [-0.004, 0.000] | 0.002 [0.000, 0.004] | 0.004 [0.002, 0.006] | 0.036 [0.031, 0.041] | 0.029 [0.020, 0.036] | -0.069 [-0.092, -0.045] |
| 19_Touvet | 0.002 [0.001, 0.004] | -0.000 [-0.001, -0.000] | -0.002 [-0.003, 0.000] | 0.001 [-0.000, 0.002] | 0.002 [0.001, 0.004] | 0.053 [0.043, 0.062] | 0.044 [0.031, 0.057] | 0.035 [0.019, 0.049] |

*Per video, shuffled AUC, within video difference from "pooled ±1 s" [95 % interval]*

| video | pooled ±1 s + motion | this frame + pooled | pooled, peaked τ = 0.1 s + motion | pooled, peaked τ = 0.25 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.3 | past 1 s + memory w = 0.3 | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.000 [0.000, 0.001] | 0.000 [-0.001, 0.002] | -0.001 [-0.003, 0.002] | 0.000 [-0.001, 0.001] | 0.000 [-0.000, 0.001] | -0.017 [-0.021, -0.013] | -0.021 [-0.026, -0.015] | -0.026 [-0.043, -0.011] |
| 2_Diner | 0.000 [-0.000, 0.001] | -0.002 [-0.003, -0.001] | -0.003 [-0.006, -0.001] | -0.001 [-0.002, 0.001] | 0.000 [-0.000, 0.001] | -0.016 [-0.021, -0.010] | -0.018 [-0.023, -0.012] | -0.033 [-0.056, -0.010] |
| 3_PlanEnergyBioLab | 0.000 [-0.000, 0.001] | 0.005 [0.003, 0.008] | 0.011 [0.006, 0.016] | 0.006 [0.003, 0.009] | 0.001 [0.001, 0.002] | -0.024 [-0.033, -0.016] | -0.026 [-0.036, -0.017] | -0.059 [-0.089, -0.031] |
| 4_Ocean (test clip) | 0.006 [0.003, 0.008] | -0.000 [-0.001, 0.001] | -0.000 [-0.004, 0.004] | 0.004 [0.001, 0.007] | 0.006 [0.003, 0.009] | -0.008 [-0.014, -0.003] | -0.018 [-0.025, -0.012] | -0.042 [-0.062, -0.022] |
| 5_Waterpark | 0.011 [0.003, 0.018] | 0.005 [0.004, 0.006] | 0.018 [0.013, 0.024] | 0.016 [0.010, 0.022] | 0.012 [0.005, 0.019] | 0.017 [0.010, 0.024] | 0.005 [-0.004, 0.013] | -0.058 [-0.089, -0.030] |
| 6_DroneFlight | 0.001 [-0.003, 0.005] | 0.000 [-0.001, 0.001] | -0.000 [-0.005, 0.004] | 0.000 [-0.004, 0.005] | 0.001 [-0.003, 0.006] | -0.003 [-0.012, 0.006] | -0.007 [-0.018, 0.005] | -0.018 [-0.047, 0.007] |
| 7_GazaFishermen (test clip) | -0.000 [-0.000, 0.000] | 0.003 [0.001, 0.004] | 0.006 [0.003, 0.010] | 0.003 [0.002, 0.005] | 0.000 [0.000, 0.001] | -0.031 [-0.046, -0.019] | -0.036 [-0.051, -0.023] | -0.038 [-0.097, 0.011] |
| 8_Sofa | 0.000 [-0.000, 0.000] | 0.001 [-0.000, 0.002] | 0.000 [-0.002, 0.003] | 0.000 [-0.001, 0.001] | 0.000 [-0.000, 0.000] | -0.018 [-0.024, -0.013] | -0.021 [-0.026, -0.014] | -0.016 [-0.040, 0.005] |
| 9_MattSwift | 0.001 [0.000, 0.001] | 0.002 [0.000, 0.003] | 0.004 [0.002, 0.006] | 0.002 [0.001, 0.003] | 0.001 [0.000, 0.001] | -0.017 [-0.022, -0.013] | -0.020 [-0.025, -0.015] | -0.025 [-0.060, 0.004] |
| 10_Cows | 0.001 [0.000, 0.001] | -0.001 [-0.002, -0.001] | -0.005 [-0.007, -0.003] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.001] | -0.027 [-0.036, -0.018] | -0.032 [-0.041, -0.024] | -0.042 [-0.060, -0.026] |
| 11_Abbottsford | 0.000 [-0.000, 0.000] | 0.001 [-0.001, 0.003] | 0.002 [-0.001, 0.006] | 0.001 [-0.000, 0.003] | 0.000 [-0.000, 0.000] | -0.014 [-0.018, -0.010] | -0.015 [-0.020, -0.011] | -0.024 [-0.042, -0.009] |
| 12_TeatroRegioTorino | 0.002 [0.001, 0.003] | 0.000 [-0.000, 0.001] | 0.001 [-0.002, 0.004] | 0.002 [0.000, 0.004] | 0.002 [0.001, 0.003] | -0.021 [-0.030, -0.013] | -0.025 [-0.034, -0.016] | -0.048 [-0.071, -0.026] |
| 13_Fountain | 0.000 [-0.001, 0.001] | 0.001 [-0.000, 0.001] | 0.001 [-0.002, 0.004] | 0.001 [-0.001, 0.002] | 0.000 [-0.000, 0.001] | -0.008 [-0.013, -0.004] | -0.010 [-0.016, -0.005] | -0.006 [-0.027, 0.011] |
| 14_Warship | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.001] | 0.000 [-0.004, 0.006] | 0.001 [-0.001, 0.005] | 0.001 [-0.000, 0.003] | -0.021 [-0.031, -0.010] | -0.024 [-0.035, -0.011] | -0.076 [-0.108, -0.048] |
| 15_Cockpit | -0.001 [-0.003, 0.001] | -0.000 [-0.001, 0.001] | -0.001 [-0.003, 0.001] | -0.001 [-0.002, 0.001] | -0.001 [-0.003, 0.001] | -0.012 [-0.018, -0.006] | -0.013 [-0.020, -0.005] | -0.030 [-0.061, 0.006] |
| 16_Turtle | 0.002 [0.000, 0.004] | -0.002 [-0.003, -0.000] | -0.003 [-0.006, 0.000] | 0.000 [-0.002, 0.003] | 0.002 [0.000, 0.004] | -0.028 [-0.041, -0.015] | -0.035 [-0.048, -0.020] | -0.254 [-0.337, -0.175] |
| 17_UnderwaterPark | 0.001 [-0.001, 0.003] | 0.001 [-0.000, 0.002] | 0.003 [0.000, 0.006] | 0.002 [0.000, 0.004] | 0.001 [-0.001, 0.003] | -0.002 [-0.007, 0.003] | -0.004 [-0.010, 0.002] | -0.019 [-0.032, -0.005] |
| 18_Bar | 0.004 [0.002, 0.007] | -0.002 [-0.003, -0.001] | -0.005 [-0.009, -0.002] | 0.001 [-0.002, 0.003] | 0.004 [0.001, 0.006] | -0.004 [-0.009, 0.002] | -0.010 [-0.017, -0.004] | -0.053 [-0.075, -0.033] |
| 19_Touvet | 0.002 [0.001, 0.003] | 0.001 [-0.000, 0.002] | 0.002 [-0.000, 0.005] | 0.003 [0.001, 0.004] | 0.002 [0.001, 0.003] | -0.002 [-0.008, 0.004] | -0.005 [-0.014, 0.003] | -0.027 [-0.054, -0.005] |

*Per video, NSS difference from "pooled ±1 s" [95 % interval]*

| video | pooled ±1 s + motion | this frame + pooled | pooled, peaked τ = 0.1 s + motion | pooled, peaked τ = 0.25 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.3 | past 1 s + memory w = 0.3 | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.00 [-0.00, 0.00] | -0.29 [-0.32, -0.25] | -0.53 [-0.58, -0.47] | -0.21 [-0.24, -0.17] | -0.02 [-0.02, -0.01] | 0.08 [0.03, 0.12] | -0.09 [-0.15, -0.03] | -0.99 [-1.33, -0.71] |
| 2_Diner | 0.01 [0.00, 0.01] | -0.33 [-0.36, -0.29] | -0.57 [-0.62, -0.52] | -0.22 [-0.25, -0.19] | -0.02 [-0.03, -0.01] | 0.11 [0.05, 0.18] | -0.04 [-0.12, 0.05] | -1.90 [-2.24, -1.55] |
| 3_PlanEnergyBioLab | 0.00 [-0.00, 0.01] | -0.15 [-0.24, -0.04] | -0.31 [-0.43, -0.16] | -0.07 [-0.14, 0.02] | 0.02 [-0.00, 0.04] | 0.09 [0.01, 0.16] | -0.10 [-0.21, -0.01] | -2.19 [-2.86, -1.65] |
| 4_Ocean (test clip) | 0.07 [0.03, 0.12] | -0.16 [-0.20, -0.12] | -0.28 [-0.34, -0.21] | -0.03 [-0.08, 0.02] | 0.07 [0.03, 0.11] | 0.06 [-0.01, 0.13] | -0.14 [-0.25, -0.05] | -1.12 [-1.49, -0.76] |
| 5_Waterpark | 0.01 [-0.16, 0.19] | -0.02 [-0.08, 0.04] | 0.07 [-0.05, 0.20] | 0.18 [0.05, 0.32] | 0.07 [-0.08, 0.24] | 0.12 [-0.02, 0.27] | -0.27 [-0.44, -0.12] | -2.58 [-3.19, -1.95] |
| 6_DroneFlight | -0.00 [-0.02, 0.01] | -0.10 [-0.12, -0.08] | -0.20 [-0.24, -0.17] | -0.09 [-0.12, -0.07] | -0.01 [-0.03, 0.00] | 0.02 [-0.01, 0.04] | -0.04 [-0.08, -0.02] | 0.29 [0.15, 0.42] |
| 7_GazaFishermen (test clip) | 0.00 [-0.00, 0.00] | -0.20 [-0.26, -0.13] | -0.37 [-0.44, -0.27] | -0.12 [-0.17, -0.05] | -0.00 [-0.02, 0.01] | 0.10 [0.05, 0.14] | -0.07 [-0.17, 0.01] | -1.27 [-2.17, -0.54] |
| 8_Sofa | -0.00 [-0.00, 0.00] | -0.32 [-0.38, -0.26] | -0.56 [-0.63, -0.49] | -0.22 [-0.26, -0.19] | -0.02 [-0.03, -0.01] | 0.06 [0.03, 0.10] | -0.10 [-0.15, -0.05] | -1.50 [-1.89, -1.12] |
| 9_MattSwift | 0.01 [-0.02, 0.04] | -0.39 [-0.43, -0.35] | -0.57 [-0.63, -0.52] | -0.19 [-0.23, -0.15] | -0.00 [-0.03, 0.03] | 0.10 [0.03, 0.16] | -0.12 [-0.21, -0.03] | -3.40 [-4.12, -2.76] |
| 10_Cows | 0.03 [0.01, 0.04] | -0.29 [-0.34, -0.25] | -0.51 [-0.57, -0.45] | -0.17 [-0.20, -0.14] | 0.01 [-0.00, 0.03] | 0.01 [-0.05, 0.06] | -0.18 [-0.26, -0.11] | -1.91 [-2.32, -1.54] |
| 11_Abbottsford | 0.00 [0.00, 0.01] | -0.35 [-0.40, -0.29] | -0.57 [-0.65, -0.50] | -0.21 [-0.25, -0.17] | -0.01 [-0.02, -0.00] | 0.09 [0.06, 0.12] | -0.10 [-0.13, -0.06] | -2.13 [-2.60, -1.71] |
| 12_TeatroRegioTorino | 0.01 [-0.04, 0.04] | -0.30 [-0.34, -0.26] | -0.52 [-0.57, -0.46] | -0.18 [-0.22, -0.15] | -0.01 [-0.05, 0.02] | 0.04 [-0.01, 0.08] | -0.14 [-0.20, -0.08] | -2.52 [-3.25, -1.82] |
| 13_Fountain | 0.00 [-0.00, 0.01] | -0.22 [-0.24, -0.20] | -0.41 [-0.45, -0.37] | -0.16 [-0.19, -0.14] | -0.01 [-0.02, -0.01] | 0.17 [0.11, 0.22] | 0.05 [-0.01, 0.11] | -0.24 [-0.48, -0.04] |
| 14_Warship | 0.01 [0.00, 0.02] | -0.28 [-0.32, -0.23] | -0.51 [-0.57, -0.44] | -0.16 [-0.20, -0.13] | -0.00 [-0.02, 0.01] | -0.01 [-0.07, 0.05] | -0.20 [-0.28, -0.11] | -2.38 [-2.73, -2.02] |
| 15_Cockpit | -0.15 [-0.23, -0.08] | -0.39 [-0.43, -0.34] | -0.65 [-0.71, -0.59] | -0.32 [-0.36, -0.29] | -0.16 [-0.23, -0.10] | -0.36 [-0.51, -0.23] | -0.57 [-0.73, -0.43] | -4.42 [-5.35, -3.51] |
| 16_Turtle | 0.06 [-0.06, 0.16] | -0.01 [-0.08, 0.06] | -0.05 [-0.14, 0.05] | 0.11 [0.04, 0.18] | 0.09 [-0.03, 0.17] | -0.64 [-1.01, -0.33] | -0.96 [-1.34, -0.63] | -7.51 [-9.46, -5.72] |
| 17_UnderwaterPark | -0.01 [-0.07, 0.03] | -0.23 [-0.28, -0.18] | -0.39 [-0.46, -0.34] | -0.15 [-0.19, -0.11] | -0.02 [-0.07, 0.02] | -0.04 [-0.13, 0.04] | -0.20 [-0.34, -0.10] | -1.30 [-2.10, -0.70] |
| 18_Bar | 0.08 [0.05, 0.11] | -0.24 [-0.27, -0.20] | -0.46 [-0.51, -0.40] | -0.13 [-0.16, -0.10] | 0.06 [0.03, 0.08] | 0.04 [-0.01, 0.09] | -0.13 [-0.18, -0.07] | -1.43 [-1.64, -1.24] |
| 19_Touvet | 0.01 [0.01, 0.02] | -0.11 [-0.15, -0.08] | -0.24 [-0.29, -0.20] | -0.09 [-0.12, -0.07] | 0.01 [-0.00, 0.01] | 0.12 [0.09, 0.15] | 0.04 [0.00, 0.08] | 0.13 [0.01, 0.22] |

*Per video, AUC-Judd*

| video | this frame | pooled ±1 s | pooled ±1 s + motion | this frame + pooled | pooled, peaked τ = 0.1 s + motion | pooled, peaked τ = 0.25 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.3 | past 1 s + memory w = 0.3 | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.680 | 0.840 | 0.841 | 0.839 | 0.834 | 0.838 | 0.841 | 0.877 | 0.869 | 0.865 |
| 2_Diner | 0.700 | 0.841 | 0.841 | 0.839 | 0.833 | 0.838 | 0.841 | 0.884 | 0.875 | 0.792 |
| 3_PlanEnergyBioLab | 0.734 | 0.844 | 0.844 | 0.844 | 0.842 | 0.844 | 0.845 | 0.883 | 0.878 | 0.875 |
| 4_Ocean (test clip) | 0.712 | 0.819 | 0.827 | 0.818 | 0.822 | 0.825 | 0.827 | 0.839 | 0.830 | 0.816 |
| 5_Waterpark | 0.749 | 0.836 | 0.853 | 0.837 | 0.854 | 0.854 | 0.853 | 0.863 | 0.848 | 0.773 |
| 6_DroneFlight | 0.578 | 0.721 | 0.728 | 0.720 | 0.725 | 0.727 | 0.728 | 0.784 | 0.774 | 0.784 |
| 7_GazaFishermen (test clip) | 0.651 | 0.791 | 0.792 | 0.791 | 0.787 | 0.790 | 0.792 | 0.838 | 0.830 | 0.794 |
| 8_Sofa | 0.647 | 0.767 | 0.767 | 0.766 | 0.761 | 0.765 | 0.768 | 0.813 | 0.807 | 0.791 |
| 9_MattSwift | 0.672 | 0.767 | 0.767 | 0.766 | 0.764 | 0.766 | 0.767 | 0.803 | 0.796 | 0.736 |
| 10_Cows | 0.717 | 0.836 | 0.836 | 0.835 | 0.831 | 0.835 | 0.836 | 0.870 | 0.863 | 0.854 |
| 11_Abbottsford | 0.698 | 0.831 | 0.831 | 0.829 | 0.824 | 0.828 | 0.831 | 0.874 | 0.867 | 0.802 |
| 12_TeatroRegioTorino | 0.685 | 0.808 | 0.810 | 0.807 | 0.806 | 0.809 | 0.810 | 0.849 | 0.842 | 0.809 |
| 13_Fountain | 0.637 | 0.795 | 0.796 | 0.793 | 0.789 | 0.794 | 0.796 | 0.844 | 0.837 | 0.834 |
| 14_Warship | 0.698 | 0.789 | 0.789 | 0.788 | 0.785 | 0.788 | 0.789 | 0.820 | 0.814 | 0.751 |
| 15_Cockpit | 0.715 | 0.820 | 0.824 | 0.819 | 0.820 | 0.823 | 0.824 | 0.858 | 0.851 | 0.703 |
| 16_Turtle | 0.825 | 0.919 | 0.921 | 0.918 | 0.919 | 0.920 | 0.921 | 0.939 | 0.932 | 0.867 |
| 17_UnderwaterPark | 0.639 | 0.743 | 0.747 | 0.743 | 0.742 | 0.745 | 0.747 | 0.774 | 0.767 | 0.753 |
| 18_Bar | 0.694 | 0.832 | 0.837 | 0.831 | 0.831 | 0.835 | 0.837 | 0.869 | 0.861 | 0.764 |
| 19_Touvet | 0.585 | 0.733 | 0.735 | 0.732 | 0.731 | 0.734 | 0.735 | 0.785 | 0.777 | 0.767 |
| all 19 videos | 0.686 | 0.807 | 0.810 | 0.806 | 0.806 | 0.809 | 0.810 | 0.846 | 0.838 | 0.796 |
| test clips 1, 4, 7 | 0.683 | 0.817 | 0.820 | 0.816 | 0.815 | 0.818 | 0.820 | 0.851 | 0.842 | 0.825 |

*Per video, shuffled AUC, within video*

| video | this frame | pooled ±1 s | pooled ±1 s + motion | this frame + pooled | pooled, peaked τ = 0.1 s + motion | pooled, peaked τ = 0.25 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.3 | past 1 s + memory w = 0.3 | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.520 | 0.529 | 0.529 | 0.529 | 0.528 | 0.529 | 0.529 | 0.512 | 0.508 | 0.503 |
| 2_Diner | 0.524 | 0.534 | 0.535 | 0.532 | 0.531 | 0.533 | 0.534 | 0.519 | 0.517 | 0.501 |
| 3_PlanEnergyBioLab | 0.576 | 0.563 | 0.563 | 0.568 | 0.573 | 0.569 | 0.564 | 0.539 | 0.536 | 0.504 |
| 4_Ocean (test clip) | 0.621 | 0.645 | 0.651 | 0.645 | 0.645 | 0.649 | 0.651 | 0.637 | 0.627 | 0.603 |
| 5_Waterpark | 0.671 | 0.681 | 0.692 | 0.686 | 0.699 | 0.697 | 0.693 | 0.698 | 0.686 | 0.623 |
| 6_DroneFlight | 0.517 | 0.535 | 0.536 | 0.535 | 0.535 | 0.535 | 0.536 | 0.532 | 0.528 | 0.517 |
| 7_GazaFishermen (test clip) | 0.541 | 0.544 | 0.544 | 0.547 | 0.550 | 0.547 | 0.544 | 0.513 | 0.507 | 0.506 |
| 8_Sofa | 0.515 | 0.516 | 0.516 | 0.517 | 0.516 | 0.516 | 0.516 | 0.498 | 0.495 | 0.500 |
| 9_MattSwift | 0.522 | 0.523 | 0.523 | 0.524 | 0.527 | 0.525 | 0.523 | 0.505 | 0.503 | 0.498 |
| 10_Cows | 0.591 | 0.604 | 0.604 | 0.602 | 0.599 | 0.603 | 0.604 | 0.577 | 0.571 | 0.562 |
| 11_Abbottsford | 0.522 | 0.521 | 0.521 | 0.522 | 0.523 | 0.522 | 0.521 | 0.507 | 0.505 | 0.497 |
| 12_TeatroRegioTorino | 0.556 | 0.577 | 0.578 | 0.577 | 0.577 | 0.578 | 0.578 | 0.556 | 0.551 | 0.528 |
| 13_Fountain | 0.507 | 0.508 | 0.508 | 0.508 | 0.509 | 0.508 | 0.508 | 0.499 | 0.498 | 0.502 |
| 14_Warship | 0.589 | 0.596 | 0.597 | 0.596 | 0.596 | 0.598 | 0.597 | 0.575 | 0.572 | 0.520 |
| 15_Cockpit | 0.532 | 0.533 | 0.532 | 0.533 | 0.532 | 0.532 | 0.532 | 0.521 | 0.520 | 0.503 |
| 16_Turtle | 0.766 | 0.801 | 0.803 | 0.799 | 0.798 | 0.801 | 0.803 | 0.773 | 0.766 | 0.547 |
| 17_UnderwaterPark | 0.527 | 0.533 | 0.534 | 0.533 | 0.536 | 0.535 | 0.534 | 0.530 | 0.528 | 0.514 |
| 18_Bar | 0.576 | 0.600 | 0.604 | 0.598 | 0.595 | 0.601 | 0.604 | 0.596 | 0.590 | 0.547 |
| 19_Touvet | 0.518 | 0.533 | 0.535 | 0.534 | 0.535 | 0.536 | 0.535 | 0.531 | 0.528 | 0.506 |
| all 19 videos | 0.564 | 0.574 | 0.575 | 0.574 | 0.575 | 0.576 | 0.576 | 0.561 | 0.556 | 0.526 |
| test clips 1, 4, 7 | 0.565 | 0.577 | 0.579 | 0.578 | 0.579 | 0.580 | 0.580 | 0.559 | 0.552 | 0.541 |

*Per video, NSS*

| video | this frame | pooled ±1 s | pooled ±1 s + motion | this frame + pooled | pooled, peaked τ = 0.1 s + motion | pooled, peaked τ = 0.25 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.3 | past 1 s + memory w = 0.3 | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 1.81 | 2.65 | 2.65 | 2.37 | 2.13 | 2.45 | 2.64 | 2.74 | 2.57 | 1.66 |
| 2_Diner | 2.18 | 3.10 | 3.10 | 2.77 | 2.53 | 2.87 | 3.08 | 3.21 | 3.06 | 1.20 |
| 3_PlanEnergyBioLab | 3.14 | 3.78 | 3.79 | 3.63 | 3.47 | 3.71 | 3.80 | 3.88 | 3.69 | 1.59 |
| 4_Ocean (test clip) | 2.11 | 2.71 | 2.78 | 2.55 | 2.43 | 2.68 | 2.78 | 2.77 | 2.56 | 1.59 |
| 5_Waterpark | 3.15 | 3.56 | 3.57 | 3.54 | 3.63 | 3.74 | 3.63 | 3.68 | 3.29 | 0.98 |
| 6_DroneFlight | 0.41 | 0.71 | 0.71 | 0.62 | 0.51 | 0.62 | 0.70 | 0.73 | 0.67 | 1.01 |
| 7_GazaFishermen (test clip) | 1.72 | 2.38 | 2.38 | 2.18 | 2.01 | 2.26 | 2.38 | 2.48 | 2.31 | 1.11 |
| 8_Sofa | 1.80 | 2.69 | 2.69 | 2.37 | 2.13 | 2.46 | 2.67 | 2.75 | 2.59 | 1.19 |
| 9_MattSwift | 3.06 | 4.10 | 4.12 | 3.72 | 3.53 | 3.91 | 4.10 | 4.20 | 3.99 | 0.70 |
| 10_Cows | 2.82 | 3.72 | 3.75 | 3.43 | 3.21 | 3.56 | 3.73 | 3.73 | 3.54 | 1.81 |
| 11_Abbottsford | 2.84 | 3.82 | 3.83 | 3.48 | 3.25 | 3.62 | 3.81 | 3.91 | 3.72 | 1.69 |
| 12_TeatroRegioTorino | 2.79 | 3.67 | 3.67 | 3.37 | 3.15 | 3.48 | 3.66 | 3.70 | 3.53 | 1.15 |
| 13_Fountain | 1.03 | 1.69 | 1.69 | 1.47 | 1.28 | 1.52 | 1.67 | 1.85 | 1.74 | 1.44 |
| 14_Warship | 2.58 | 3.51 | 3.52 | 3.23 | 3.00 | 3.34 | 3.50 | 3.50 | 3.31 | 1.13 |
| 15_Cockpit | 4.13 | 5.15 | 5.00 | 4.77 | 4.50 | 4.83 | 4.99 | 4.79 | 4.58 | 0.73 |
| 16_Turtle | 8.74 | 9.14 | 9.20 | 9.13 | 9.09 | 9.25 | 9.23 | 8.50 | 8.18 | 1.63 |
| 17_UnderwaterPark | 1.82 | 2.48 | 2.47 | 2.25 | 2.08 | 2.33 | 2.46 | 2.44 | 2.28 | 1.18 |
| 18_Bar | 1.64 | 2.42 | 2.50 | 2.18 | 1.96 | 2.29 | 2.48 | 2.46 | 2.30 | 0.99 |
| 19_Touvet | 0.49 | 0.87 | 0.89 | 0.76 | 0.63 | 0.78 | 0.88 | 0.99 | 0.91 | 1.00 |
| all 19 videos | 2.57 | 3.29 | 3.30 | 3.07 | 2.90 | 3.17 | 3.30 | 3.30 | 3.12 | 1.25 |
| test clips 1, 4, 7 | 1.89 | 2.59 | 2.62 | 2.38 | 2.20 | 2.47 | 2.61 | 2.67 | 2.49 | 1.46 |

*Per video, kept*

| video | this frame | pooled ±1 s | pooled ±1 s + motion | this frame + pooled | pooled, peaked τ = 0.1 s + motion | pooled, peaked τ = 0.25 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.3 | past 1 s + memory w = 0.3 | SalFormer360-ft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_PortoRiverside (test clip) | 0.964 | 0.558 | 0.558 | 0.834 | 0.830 | 0.695 | 0.595 | 0.663 | 0.645 | 0.931 |
| 2_Diner | 0.972 | 0.589 | 0.589 | 0.820 | 0.821 | 0.691 | 0.613 | 0.678 | 0.664 | 0.959 |
| 3_PlanEnergyBioLab | 0.968 | 0.530 | 0.530 | 0.787 | 0.799 | 0.646 | 0.560 | 0.622 | 0.620 | 0.986 |
| 4_Ocean (test clip) | 0.972 | 0.681 | 0.684 | 0.876 | 0.870 | 0.782 | 0.717 | 0.782 | 0.727 | 0.826 |
| 5_Waterpark | 0.978 | 0.648 | 0.605 | 0.850 | 0.811 | 0.718 | 0.643 | 0.739 | 0.725 | 0.821 |
| 6_DroneFlight | 0.962 | 0.797 | 0.825 | 0.942 | 0.928 | 0.891 | 0.857 | 0.907 | 0.813 | 0.950 |
| 7_GazaFishermen (test clip) | 0.964 | 0.559 | 0.560 | 0.845 | 0.829 | 0.702 | 0.602 | 0.645 | 0.624 | 0.782 |
| 8_Sofa | 0.973 | 0.465 | 0.466 | 0.763 | 0.778 | 0.607 | 0.501 | 0.553 | 0.553 | 0.848 |
| 9_MattSwift | 0.975 | 0.333 | 0.332 | 0.635 | 0.651 | 0.440 | 0.351 | 0.388 | 0.416 | 0.678 |
| 10_Cows | 0.965 | 0.541 | 0.538 | 0.784 | 0.787 | 0.641 | 0.564 | 0.664 | 0.657 | 0.874 |
| 11_Abbottsford | 0.972 | 0.463 | 0.463 | 0.725 | 0.732 | 0.569 | 0.487 | 0.519 | 0.529 | 0.824 |
| 12_TeatroRegioTorino | 0.976 | 0.486 | 0.484 | 0.738 | 0.761 | 0.601 | 0.511 | 0.572 | 0.576 | 0.969 |
| 13_Fountain | 0.970 | 0.671 | 0.673 | 0.894 | 0.876 | 0.791 | 0.715 | 0.804 | 0.757 | 0.875 |
| 14_Warship | 0.966 | 0.512 | 0.510 | 0.779 | 0.784 | 0.627 | 0.541 | 0.634 | 0.631 | 0.913 |
| 15_Cockpit | 0.967 | 0.450 | 0.463 | 0.675 | 0.725 | 0.558 | 0.481 | 0.587 | 0.618 | 0.857 |
| 16_Turtle | 0.977 | 0.736 | 0.715 | 0.840 | 0.827 | 0.756 | 0.728 | 0.798 | 0.767 | 0.939 |
| 17_UnderwaterPark | 0.972 | 0.511 | 0.512 | 0.801 | 0.811 | 0.656 | 0.551 | 0.628 | 0.597 | 0.935 |
| 18_Bar | 0.964 | 0.682 | 0.678 | 0.890 | 0.881 | 0.791 | 0.716 | 0.803 | 0.753 | 0.950 |
| 19_Touvet | 0.969 | 0.779 | 0.779 | 0.941 | 0.915 | 0.867 | 0.817 | 0.882 | 0.801 | 0.856 |
| all 19 videos | 0.970 | 0.580 | 0.578 | 0.812 | 0.811 | 0.686 | 0.609 | 0.678 | 0.657 | 0.881 |
| test clips 1, 4, 7 | 0.967 | 0.604 | 0.606 | 0.853 | 0.845 | 0.730 | 0.643 | 0.702 | 0.669 | 0.845 |

*Example frames (numbers only; maps from one half of the people, split 1)*

13_Fountain, frame 385: adding the memory map catches the other half's fixations that the same recipe without memory (pooled, peaked τ = 0.25 s + motion) misses.

| map | kept | held-out hit rate | split 1: other half's fixations inside | AUC-Judd | NSS |
|---|---:|---:|---:|---:|---:|
| this frame | 0.963 | 0.254 | 11 of 38 | 0.674 | 1.38 |
| this frame, wider Gaussian | 1.000 | 0.333 | 12 of 38 | 0.688 | 1.51 |
| pooled ±1 s | 0.695 | 0.351 | 13 of 38 | 0.806 | 1.95 |
| pooled ±1 s + motion | 0.687 | 0.363 | 13 of 38 | 0.810 | 1.93 |
| past 1 s + motion | 0.672 | 0.335 | 12 of 38 | 0.763 | 1.61 |
| this frame + pooled | 0.907 | 0.288 | 11 of 38 | 0.805 | 1.79 |
| pooled, peaked τ = 0.1 s + motion | 0.898 | 0.261 | 11 of 38 | 0.804 | 1.63 |
| pooled, peaked τ = 0.25 s + motion | 0.813 | 0.313 | 12 of 38 | 0.808 | 1.84 |
| pooled, peaked τ = 0.5 s + motion | 0.722 | 0.360 | 13 of 38 | 0.810 | 1.94 |
| pooled + memory w = 0.3 | 0.839 | 0.491 | 16 of 38 | 0.850 | 2.28 |
| past 1 s + memory w = 0.3 | 0.786 | 0.470 | 16 of 38 | 0.845 | 2.15 |
| SalFormer360-ft | 0.820 | 0.835 | 33 of 38 | 0.813 | 1.30 |

9_MattSwift, frame 100: peaked weighting keeps far more of the dataset map than flat pooling.

| map | kept | held-out hit rate | split 1: other half's fixations inside | AUC-Judd | NSS |
|---|---:|---:|---:|---:|---:|
| this frame | 0.956 | 0.114 | 6 of 42 | 0.573 | 0.91 |
| this frame, wider Gaussian | 0.956 | 0.120 | 6 of 42 | 0.575 | 0.91 |
| pooled ±1 s | 0.201 | 0.165 | 6 of 42 | 0.676 | 1.91 |
| pooled ±1 s + motion | 0.207 | 0.165 | 6 of 42 | 0.676 | 1.90 |
| past 1 s + motion | 0.256 | 0.169 | 6 of 42 | 0.669 | 1.86 |
| this frame + pooled | 0.687 | 0.134 | 6 of 42 | 0.675 | 1.42 |
| pooled, peaked τ = 0.1 s + motion | 0.796 | 0.135 | 6 of 42 | 0.667 | 1.15 |
| pooled, peaked τ = 0.25 s + motion | 0.422 | 0.159 | 6 of 42 | 0.673 | 1.58 |
| pooled, peaked τ = 0.5 s + motion | 0.240 | 0.160 | 6 of 42 | 0.677 | 1.84 |
| pooled + memory w = 0.3 | 0.332 | 0.158 | 6 of 42 | 0.695 | 1.79 |
| past 1 s + memory w = 0.3 | 0.365 | 0.171 | 6 of 42 | 0.702 | 1.77 |
| SalFormer360-ft | 0.686 | 0.701 | 30 of 42 | 0.746 | 0.53 |

16_Turtle, frame 440: the frame where motion compensation failed in the first figure.

| map | kept | held-out hit rate | split 1: other half's fixations inside | AUC-Judd | NSS |
|---|---:|---:|---:|---:|---:|
| this frame | 0.976 | 0.798 | 43 of 53 | 0.927 | 18.19 |
| this frame, wider Gaussian | 0.976 | 0.802 | 43 of 53 | 0.928 | 18.16 |
| pooled ±1 s | 0.904 | 0.793 | 42 of 53 | 0.956 | 18.06 |
| pooled ±1 s + motion | 0.952 | 0.806 | 44 of 53 | 0.957 | 17.54 |
| past 1 s + motion | 0.963 | 0.817 | 45 of 53 | 0.953 | 15.50 |
| this frame + pooled | 0.940 | 0.796 | 43 of 53 | 0.956 | 18.20 |
| pooled, peaked τ = 0.1 s + motion | 0.940 | 0.814 | 44 of 53 | 0.958 | 18.05 |
| pooled, peaked τ = 0.25 s + motion | 0.976 | 0.808 | 44 of 53 | 0.957 | 17.80 |
| pooled, peaked τ = 0.5 s + motion | 0.952 | 0.806 | 44 of 53 | 0.956 | 17.60 |
| pooled + memory w = 0.3 | 0.976 | 0.808 | 44 of 53 | 0.972 | 15.22 |
| past 1 s + memory w = 0.3 | 0.988 | 0.811 | 44 of 53 | 0.973 | 14.76 |
| SalFormer360-ft | 0.821 | 0.905 | 47 of 53 | 0.750 | 0.25 |

**Conclusions.**

- Weighting the current frame more brings "kept" up to 0.81 at most (τ = 0.1 s), and at that point the held-out hit rate and NSS are clearly worse than "pooled ±1 s". Only τ = 0.5 s holds the held-out scores, and it keeps 0.61. Optimising "kept" works against predicting other people, so it was dropped as a goal.
- Scene memory gives the best AUC-Judd: 0.846 against 0.807 for "pooled ±1 s", higher on all 19 videos with every per-video interval above 0. The live version ("past 1 s + memory") does the same. On the within-video shuffled AUC it is worse on 18 of 19 videos, which is expected: that measure cancels the video's static salient objects, which is what memory adds (section 6 checks this).
- The first 3 s of one half predict the other half's first 3 s well, and the places looked at in the first 3 s keep being looked at later (21–26 % of fixations in about 3 % of the sphere). A recent 3 s window catches more in total in a somewhat larger area; per unit of area the two are about the same.
- "this frame + pooled" adds the least area for what it catches (11.9 × chance per added area) but ranks slightly below "pooled ±1 s" on AUC-Judd and NSS.

## 6. Shuffled AUC with comparison fixations from other videos (report 5)

Comparison fixations for "across videos": 334 from each of the other 18 videos of the same set (about 6,000), spread evenly over all their fixations, any time. "Within video" as in section 5: fixations of the same video more than 2 s away, at the evaluated frames. Both scored on the same held-out fixations of the same frames and splits, recomputed in this run.

**Salient360** (19 videos, 1887 frames, held out as in section 2):

| map | shuffled AUC, across videos | shuffled AUC, within video |
|---|---:|---:|
| pooled ±1 s | 0.730 | 0.574 |
| pooled ±1 s + motion | 0.732 | 0.575 |
| pooled, peaked τ = 0.5 s + motion | 0.732 | 0.576 |
| pooled + memory w = 0.1 | 0.742 | 0.566 |
| pooled + memory w = 0.3 | 0.743 | 0.561 |
| past 1 s + memory w = 0.1 | 0.734 | 0.561 |
| past 1 s + memory w = 0.3 | 0.735 | 0.556 |
| SalFormer360-ft | 0.625 | 0.526 |
| average fixation map of this video | 0.743 | 0.462 |

*Significance against "pooled ±1 s", shuffled AUC across videos*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| pooled ±1 s + motion | 0.002 | 17 / 2 | 9 / 0 | < 0.001 |
| pooled, peaked τ = 0.5 s + motion | 0.002 | 18 / 1 | 11 / 0 | < 0.001 |
| pooled + memory w = 0.1 | 0.012 | 17 / 2 | 15 / 0 | < 0.001 |
| pooled + memory w = 0.3 | 0.013 | 16 / 3 | 14 / 1 | 0.004 |
| past 1 s + memory w = 0.1 | 0.004 | 13 / 6 | 7 / 3 | 0.167 |
| past 1 s + memory w = 0.3 | 0.006 | 14 / 5 | 8 / 3 | 0.064 |
| SalFormer360-ft | -0.103 | 0 / 19 | 0 / 19 | < 0.001 |
| average fixation map of this video | 0.014 | 14 / 5 | 12 / 3 | 0.064 |

*Significance against "pooled ±1 s", shuffled AUC within video*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| pooled ±1 s + motion | 0.002 | 17 / 2 | 10 / 0 | < 0.001 |
| pooled, peaked τ = 0.5 s + motion | 0.002 | 18 / 1 | 10 / 0 | < 0.001 |
| pooled + memory w = 0.1 | -0.008 | 1 / 18 | 1 / 13 | < 0.001 |
| pooled + memory w = 0.3 | -0.014 | 1 / 18 | 1 / 14 | < 0.001 |
| past 1 s + memory w = 0.1 | -0.013 | 1 / 18 | 0 / 15 | < 0.001 |
| past 1 s + memory w = 0.3 | -0.018 | 1 / 18 | 0 / 15 | < 0.001 |
| SalFormer360-ft | -0.047 | 0 / 19 | 0 / 13 | < 0.001 |
| average fixation map of this video | -0.110 | 0 / 19 | 0 / 19 | < 0.001 |

*Per video, shuffled AUC across videos*

| video | pooled ±1 s | pooled ±1 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.1 | pooled + memory w = 0.3 | past 1 s + memory w = 0.1 | past 1 s + memory w = 0.3 | SalFormer360-ft | average fixation map of this video |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10_Cows | 0.752 | 0.752 | 0.753 | 0.760 | 0.761 | 0.752 | 0.753 | 0.700 | 0.761 |
| 11_Abbottsford | 0.784 | 0.784 | 0.784 | 0.809 | 0.812 | 0.802 | 0.805 | 0.629 | 0.832 |
| 12_TeatroRegioTorino | 0.714 | 0.716 | 0.716 | 0.720 | 0.719 | 0.713 | 0.712 | 0.607 | 0.707 |
| 13_Fountain | 0.689 | 0.690 | 0.690 | 0.712 | 0.718 | 0.706 | 0.712 | 0.649 | 0.734 |
| 14_Warship | 0.741 | 0.742 | 0.742 | 0.749 | 0.748 | 0.742 | 0.742 | 0.658 | 0.752 |
| 15_Cockpit | 0.750 | 0.752 | 0.752 | 0.761 | 0.760 | 0.753 | 0.754 | 0.536 | 0.783 |
| 16_Turtle | 0.876 | 0.878 | 0.878 | 0.877 | 0.872 | 0.869 | 0.865 | 0.684 | 0.849 |
| 17_UnderwaterPark | 0.692 | 0.694 | 0.695 | 0.687 | 0.685 | 0.676 | 0.675 | 0.632 | 0.708 |
| 18_Bar | 0.719 | 0.723 | 0.723 | 0.731 | 0.733 | 0.724 | 0.725 | 0.649 | 0.709 |
| 19_Touvet | 0.602 | 0.604 | 0.604 | 0.620 | 0.623 | 0.616 | 0.619 | 0.537 | 0.622 |
| 1_PortoRiverside | 0.734 | 0.735 | 0.735 | 0.745 | 0.746 | 0.736 | 0.738 | 0.673 | 0.760 |
| 2_Diner | 0.780 | 0.780 | 0.780 | 0.795 | 0.799 | 0.784 | 0.789 | 0.565 | 0.816 |
| 3_PlanEnergyBioLab | 0.747 | 0.747 | 0.747 | 0.761 | 0.763 | 0.757 | 0.759 | 0.683 | 0.766 |
| 4_Ocean | 0.786 | 0.792 | 0.792 | 0.785 | 0.784 | 0.774 | 0.773 | 0.732 | 0.779 |
| 5_Waterpark | 0.761 | 0.770 | 0.771 | 0.777 | 0.776 | 0.765 | 0.764 | 0.651 | 0.691 |
| 6_DroneFlight | 0.632 | 0.635 | 0.635 | 0.662 | 0.663 | 0.654 | 0.655 | 0.588 | 0.670 |
| 7_GazaFishermen | 0.697 | 0.697 | 0.698 | 0.714 | 0.717 | 0.707 | 0.710 | 0.562 | 0.727 |
| 8_Sofa | 0.686 | 0.686 | 0.686 | 0.704 | 0.707 | 0.699 | 0.702 | 0.623 | 0.729 |
| 9_MattSwift | 0.706 | 0.706 | 0.706 | 0.710 | 0.711 | 0.704 | 0.706 | 0.532 | 0.718 |

**VR-EyeTracking training videos (σ 3°)** (9 videos, 864 frames, held out as in section 2):

| map | shuffled AUC, across videos | shuffled AUC, within video |
|---|---:|---:|
| pooled ±1 s | 0.784 | 0.688 |
| pooled ±1 s + motion | 0.784 | 0.689 |
| pooled, peaked τ = 0.5 s + motion | 0.784 | 0.690 |
| pooled + memory w = 0.1 | 0.787 | 0.672 |
| pooled + memory w = 0.3 | 0.782 | 0.660 |
| past 1 s + memory w = 0.1 | 0.777 | 0.663 |
| past 1 s + memory w = 0.3 | 0.772 | 0.651 |
| average fixation map of this video | 0.722 | 0.466 |

*Significance against "pooled ±1 s", shuffled AUC across videos*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| pooled ±1 s + motion | 0.000 | 5 / 4 | 4 / 1 | 1.000 |
| pooled, peaked τ = 0.5 s + motion | 0.000 | 6 / 3 | 4 / 1 | 0.508 |
| pooled + memory w = 0.1 | 0.002 | 6 / 3 | 3 / 2 | 0.508 |
| pooled + memory w = 0.3 | -0.003 | 4 / 5 | 2 / 3 | 1.000 |
| past 1 s + memory w = 0.1 | -0.008 | 3 / 6 | 1 / 4 | 0.508 |
| past 1 s + memory w = 0.3 | -0.013 | 3 / 6 | 1 / 4 | 0.508 |
| average fixation map of this video | -0.061 | 1 / 8 | 1 / 6 | 0.039 |

*Significance against "pooled ±1 s", shuffled AUC within video*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| pooled ±1 s + motion | 0.001 | 7 / 2 | 4 / 0 | 0.180 |
| pooled, peaked τ = 0.5 s + motion | 0.001 | 7 / 2 | 4 / 0 | 0.180 |
| pooled + memory w = 0.1 | -0.016 | 0 / 9 | 0 / 8 | 0.004 |
| pooled + memory w = 0.3 | -0.029 | 0 / 9 | 0 / 8 | 0.004 |
| past 1 s + memory w = 0.1 | -0.025 | 0 / 9 | 0 / 9 | 0.004 |
| past 1 s + memory w = 0.3 | -0.038 | 0 / 9 | 0 / 9 | 0.004 |
| average fixation map of this video | -0.218 | 0 / 9 | 0 / 9 | 0.004 |

*Per video, shuffled AUC across videos*

| video | pooled ±1 s | pooled ±1 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.1 | pooled + memory w = 0.3 | past 1 s + memory w = 0.1 | past 1 s + memory w = 0.3 | average fixation map of this video |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vret_001 | 0.722 | 0.722 | 0.722 | 0.713 | 0.702 | 0.701 | 0.690 | 0.671 |
| vret_043 | 0.736 | 0.736 | 0.736 | 0.746 | 0.743 | 0.740 | 0.736 | 0.671 |
| vret_072 | 0.894 | 0.895 | 0.895 | 0.895 | 0.892 | 0.889 | 0.887 | 0.865 |
| vret_088 | 0.789 | 0.795 | 0.795 | 0.771 | 0.758 | 0.748 | 0.737 | 0.583 |
| vret_116 | 0.728 | 0.729 | 0.729 | 0.730 | 0.725 | 0.721 | 0.717 | 0.687 |
| vret_130 | 0.908 | 0.908 | 0.908 | 0.902 | 0.893 | 0.892 | 0.882 | 0.726 |
| vret_159 | 0.729 | 0.729 | 0.729 | 0.740 | 0.740 | 0.731 | 0.731 | 0.716 |
| vret_187 | 0.732 | 0.725 | 0.725 | 0.739 | 0.734 | 0.725 | 0.720 | 0.728 |
| vret_202 | 0.781 | 0.781 | 0.781 | 0.800 | 0.802 | 0.795 | 0.797 | 0.819 |

**YT360 training videos (σ 3°)** (10 videos, 840 frames, held out as in section 2):

| map | shuffled AUC, across videos | shuffled AUC, within video |
|---|---:|---:|
| pooled ±1 s | 0.820 | 0.604 |
| pooled ±1 s + motion | 0.820 | 0.606 |
| pooled, peaked τ = 0.5 s + motion | 0.820 | 0.607 |
| pooled + memory w = 0.1 | 0.822 | 0.603 |
| pooled + memory w = 0.3 | 0.822 | 0.599 |
| past 1 s + memory w = 0.1 | 0.815 | 0.599 |
| past 1 s + memory w = 0.3 | 0.815 | 0.594 |
| average fixation map of this video | 0.805 | 0.484 |

*Significance against "pooled ±1 s", shuffled AUC across videos*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| pooled ±1 s + motion | 0.000 | 8 / 2 | 4 / 1 | 0.109 |
| pooled, peaked τ = 0.5 s + motion | 0.000 | 8 / 2 | 3 / 1 | 0.109 |
| pooled + memory w = 0.1 | 0.002 | 8 / 2 | 4 / 1 | 0.109 |
| pooled + memory w = 0.3 | 0.002 | 6 / 4 | 3 / 1 | 0.754 |
| past 1 s + memory w = 0.1 | -0.005 | 2 / 8 | 0 / 4 | 0.109 |
| past 1 s + memory w = 0.3 | -0.005 | 4 / 6 | 1 / 3 | 0.754 |
| average fixation map of this video | -0.015 | 5 / 5 | 2 / 5 | 1.000 |

*Significance against "pooled ±1 s", shuffled AUC within video*

| map | mean difference per video | videos better / worse | interval above 0 / below 0 | sign test p |
|---|---:|---:|---:|---:|
| pooled ±1 s + motion | 0.002 | 9 / 1 | 5 / 0 | 0.021 |
| pooled, peaked τ = 0.5 s + motion | 0.002 | 10 / 0 | 8 / 0 | 0.002 |
| pooled + memory w = 0.1 | -0.001 | 2 / 8 | 1 / 4 | 0.109 |
| pooled + memory w = 0.3 | -0.006 | 1 / 9 | 1 / 8 | 0.021 |
| past 1 s + memory w = 0.1 | -0.006 | 2 / 8 | 0 / 7 | 0.109 |
| past 1 s + memory w = 0.3 | -0.010 | 1 / 9 | 0 / 8 | 0.021 |
| average fixation map of this video | -0.120 | 0 / 10 | 0 / 10 | 0.002 |

*Per video, shuffled AUC across videos*

| video | pooled ±1 s | pooled ±1 s + motion | pooled, peaked τ = 0.5 s + motion | pooled + memory w = 0.1 | pooled + memory w = 0.3 | past 1 s + memory w = 0.1 | past 1 s + memory w = 0.3 | average fixation map of this video |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yt360_-ZrvrKMC7Tc-28 | 0.740 | 0.742 | 0.742 | 0.732 | 0.724 | 0.718 | 0.711 | 0.647 |
| yt360_Aa2muAMdV5s-156 | 0.721 | 0.724 | 0.724 | 0.729 | 0.733 | 0.720 | 0.724 | 0.697 |
| yt360_KJYHP1ahjQ4-141 | 0.801 | 0.802 | 0.802 | 0.805 | 0.806 | 0.801 | 0.803 | 0.813 |
| yt360_TLcg-0-Qtw4-519 | 0.804 | 0.804 | 0.804 | 0.809 | 0.811 | 0.806 | 0.808 | 0.809 |
| yt360_XZY3nPnDCQw-4951 | 0.841 | 0.841 | 0.841 | 0.843 | 0.844 | 0.833 | 0.836 | 0.857 |
| yt360_dQaNzwqCUAA-281 | 0.816 | 0.816 | 0.816 | 0.820 | 0.821 | 0.816 | 0.817 | 0.818 |
| yt360_iDy7xYWMu1k-367 | 0.829 | 0.829 | 0.829 | 0.830 | 0.827 | 0.823 | 0.821 | 0.812 |
| yt360_neNVkg7E7ic-394 | 0.946 | 0.941 | 0.941 | 0.946 | 0.945 | 0.933 | 0.932 | 0.908 |
| yt360_slR85J2Ua28-173 | 0.871 | 0.870 | 0.871 | 0.872 | 0.870 | 0.869 | 0.867 | 0.855 |
| yt360_xyccHcWoS74-138 | 0.830 | 0.831 | 0.831 | 0.835 | 0.836 | 0.826 | 0.828 | 0.832 |

**Conclusions.**

- **The across-video measure does what it should.** The control map, all of one half's fixations over the whole video, scores well above 0.5 across videos (0.743 on Salient360) and below 0.5 within video (0.462): it knows the video's salient objects but nothing about when they are looked at. Gaze-based maps score far higher across videos than within (0.73 against 0.57 for "pooled ±1 s" on Salient360).
- **Scene memory, judged fairly, is a small gain on Salient360 and none on the training videos.** On Salient360, "pooled + memory" is 0.012–0.013 above "pooled ±1 s" across videos, better on 16–17 of 19 videos with the interval above 0 on 14–15. It is also about level with the whole-video average map. On the 9 VR-EyeTracking and 10 YT360 training videos the differences are within ±0.003 and not significant. The live version ("past 1 s + memory") is no better than "pooled ±1 s" on any set. The large AUC-Judd gain of section 5 therefore comes mostly from knowing which parts of the scene draw the eye at all, which a map of the whole video knows as well.
- **SalFormer360-ft scores 0.625 across videos.** That is above chance, so the within-video score below 0.5 was the measure cancelling its static predictions. It is still 0.10 below "pooled ±1 s" on all 19 videos and below the whole-video average map: it predicts the scene's salient objects less well than the people who watched it do.
- **Motion and peaked weighting:** both are a hair above "pooled ±1 s" on Salient360 (+0.002, 17–18 of 19 videos) and level on the training videos.

## 7. Training videos: VR-EyeTracking and YT360

10 VR-EyeTracking and 10 YT360 videos that have published "mc3" training maps were picked evenly from the sorted list. VR-EyeTracking 016 cannot be decoded by Chrome (section 9), so 9 + 10 are used. σ 3° for every map, to match the published maps. Every 10th frame of the published maps, first and last second left out. Kept, added and correlation are measured against the published map at its own size (128 × 64); held-out scores at 256 × 128. The published maps are built from everyone, so they cannot be scored on held-out people; OverGaze's "pooled ±1 s + motion" is the same recipe rebuilt from one half. VR-EyeTracking gaze is raw samples (one per person per frame, speed filter 75°/s), so "this frame" is very sparse there. YT360 gives fixations; a person who appears in several viewing conditions stays in one half.

| video | people | frames | correlation of "pooled ±1 s + motion" with the published map | kept: this frame | kept: pooled ±1 s | kept: pooled ±1 s + motion | kept: this frame + pooled | kept: pooled + memory w = 0.3 | AUC-Judd: this frame | AUC-Judd: pooled ±1 s | AUC-Judd: pooled ±1 s + motion | AUC-Judd: this frame + pooled | AUC-Judd: pooled + memory w = 0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vret_001 | 33 | 44 | 0.992 | 0.475 | 0.937 | 0.938 | 0.706 | 0.880 | 0.692 | 0.861 | 0.861 | 0.860 | 0.893 |
| vret_043 | 30 | 166 | 0.999 | 0.669 | 0.979 | 0.973 | 0.815 | 0.925 | 0.745 | 0.852 | 0.852 | 0.850 | 0.906 |
| vret_072 | 33 | 111 | 0.999 | 0.775 | 0.971 | 0.962 | 0.907 | 0.947 | 0.877 | 0.926 | 0.926 | 0.926 | 0.954 |
| vret_088 | 34 | 106 | 0.963 | 0.556 | 0.743 | 0.821 | 0.686 | 0.772 | 0.758 | 0.867 | 0.870 | 0.867 | 0.879 |
| vret_116 | 34 | 61 | 0.995 | 0.653 | 0.900 | 0.924 | 0.772 | 0.875 | 0.758 | 0.876 | 0.877 | 0.875 | 0.910 |
| vret_130 | 34 | 88 | 0.995 | 0.641 | 0.910 | 0.929 | 0.808 | 0.907 | 0.904 | 0.955 | 0.955 | 0.955 | 0.968 |
| vret_159 | 32 | 65 | 1.000 | 0.748 | 0.950 | 0.955 | 0.842 | 0.872 | 0.743 | 0.847 | 0.847 | 0.846 | 0.889 |
| vret_187 | 30 | 112 | 0.987 | 0.519 | 0.752 | 0.901 | 0.641 | 0.832 | 0.636 | 0.748 | 0.747 | 0.748 | 0.771 |
| vret_202 | 28 | 111 | 0.996 | 0.646 | 0.946 | 0.949 | 0.795 | 0.941 | 0.714 | 0.873 | 0.873 | 0.872 | 0.919 |
| yt360_-ZrvrKMC7Tc-28 | 101 | 84 | 0.990 | 0.567 | 0.828 | 0.904 | 0.738 | 0.867 | 0.755 | 0.866 | 0.868 | 0.864 | 0.871 |
| yt360_Aa2muAMdV5s-156 | 100 | 84 | 0.994 | 0.641 | 0.913 | 0.910 | 0.793 | 0.850 | 0.738 | 0.863 | 0.864 | 0.860 | 0.889 |
| yt360_KJYHP1ahjQ4-141 | 100 | 84 | 0.999 | 0.814 | 0.949 | 0.938 | 0.891 | 0.911 | 0.833 | 0.915 | 0.915 | 0.913 | 0.934 |
| yt360_TLcg-0-Qtw4-519 | 101 | 84 | 1.000 | 0.852 | 0.965 | 0.965 | 0.914 | 0.964 | 0.852 | 0.925 | 0.925 | 0.923 | 0.944 |
| yt360_XZY3nPnDCQw-4951 | 99 | 84 | 0.994 | 0.766 | 0.858 | 0.868 | 0.816 | 0.920 | 0.793 | 0.913 | 0.913 | 0.909 | 0.932 |
| yt360_dQaNzwqCUAA-281 | 100 | 84 | 1.000 | 0.910 | 0.983 | 0.984 | 0.952 | 0.969 | 0.851 | 0.907 | 0.907 | 0.905 | 0.929 |
| yt360_iDy7xYWMu1k-367 | 99 | 84 | 0.997 | 0.864 | 0.933 | 0.946 | 0.907 | 0.954 | 0.852 | 0.920 | 0.920 | 0.919 | 0.934 |
| yt360_neNVkg7E7ic-394 | 101 | 84 | 0.935 | 0.504 | 0.640 | 0.680 | 0.604 | 0.670 | 0.933 | 0.969 | 0.971 | 0.969 | 0.978 |
| yt360_slR85J2Ua28-173 | 101 | 84 | 0.994 | 0.808 | 0.896 | 0.893 | 0.853 | 0.955 | 0.900 | 0.954 | 0.954 | 0.953 | 0.966 |
| yt360_xyccHcWoS74-138 | 101 | 84 | 0.999 | 0.689 | 0.959 | 0.953 | 0.828 | 0.914 | 0.810 | 0.928 | 0.929 | 0.925 | 0.945 |
| VR-EyeTracking, 9 videos |  | 864 | 0.992 | 0.638 | 0.898 | 0.930 | 0.779 | 0.887 | 0.760 | 0.865 | 0.865 | 0.864 | 0.898 |
| YT360, 10 videos |  | 840 | 0.990 | 0.741 | 0.892 | 0.904 | 0.830 | 0.897 | 0.832 | 0.916 | 0.917 | 0.914 | 0.932 |

| set | map | added, × published area | held-out hit rate | shuffled AUC, within video | NSS |
|---|---:|---:|---:|---:|---:|
| VR-EyeTracking | this frame | 0.30 | 0.412 | 0.665 | 4.48 |
| VR-EyeTracking | pooled ±1 s | 0.07 | 0.501 | 0.688 | 4.92 |
| VR-EyeTracking | pooled ±1 s + motion | 0.03 | 0.505 | 0.690 | 4.96 |
| VR-EyeTracking | this frame + pooled | 0.18 | 0.472 | 0.689 | 4.86 |
| VR-EyeTracking | pooled + memory w = 0.3 | 0.35 | 0.528 | 0.660 | 4.86 |
| YT360 | this frame | 0.28 | 0.540 | 0.600 | 7.08 |
| YT360 | pooled ±1 s | 0.04 | 0.590 | 0.605 | 7.51 |
| YT360 | pooled ±1 s + motion | 0.02 | 0.589 | 0.606 | 7.50 |
| YT360 | this frame + pooled | 0.13 | 0.576 | 0.607 | 7.44 |
| YT360 | pooled + memory w = 0.3 | 0.23 | 0.603 | 0.599 | 7.36 |

**Conclusions.** OverGaze rebuilds the published training maps (correlation 0.99 on both sets), and the published maps stay as they are. On held-out people the training videos show the same pattern as Salient360: scene memory has the best AUC-Judd on both sets (0.898 against 0.865 on VR-EyeTracking, 0.932 against 0.917 on YT360) and a lower within-video shuffled AUC, while across videos it is level with "pooled ±1 s" (section 6); "this frame + pooled" is about equal to "pooled ±1 s + motion".

## 8. Earlier reports that were replaced

- **First figure (temporal pooling against the dataset map).** OverGaze then fell back to 30 frames per second when its frame-rate measurement failed on a busy computer; 18 of the 19 runs used 30, which misaligned the frame-numbered Salient360 fixations on the 25 and 24 frames-per-second videos (and slightly on the 29.97 ones). Its motion compensation used the browser's built-in estimate before the fix. Replaced. The frame rate is now read from the file header (pull request #7).
- **Second report (correlation with the dataset map and with the other half of the people).** True frame rates were set by hand, but motion compensation still used the built-in estimate before the fix, and correlation turned out to be the wrong measure for pooled maps (it rewards matching the same people at the same moment). Replaced by report 3.
- **Section 1 table** (the first check against the published training maps) used the built-in motion estimate before the fix; it is kept because section 7 repeats the check with OpenCV motion and agrees.
- **Report 3** used OpenCV motion for "+ motion" and showed the built-in estimate before and after the fix as separate rows, both run outside the browser on the frames OpenCV decoded. Its numbers are kept in section 4.
- **Reports 3 and 4 were run before the test-browser fix** (section 9). Each run checked its observer count after loading the gaze (report 4 also after loading the motion) and was repeated when the check failed (16_Turtle and 4_Ocean in report 3, 14_Warship in report 4). The maps both reports share give the same averages (for example "pooled ±1 s": kept 0.580, hit rate 0.356, AUC-Judd 0.807 in both), and "this frame" keeps about 0.97 of the dataset map on every video. The memory-baseline runs of report 4 and all of report 5 and section 7 were run after the fix.

## 9. Faults found along the way

- **Doubled observers** in parallel runs were a fault in the test-browser driver, not in OverGaze: it picked the debugging port at random, two browsers could get the same one, and two scripts then drove one page (one caught case: 6_DroneFlight's page also held 13_Fountain's files). Fixed in pull request #10: Chrome now picks a free port itself.
- **Frame rate** fell back to 30 frames per second without saying so when the measurement failed on a busy computer. Fixed in pull request #7.
- **Built-in motion estimate** spread movement into still areas. Fixed in pull request #8; OpenCV motion files can now be loaded.
- **Dataset files:** VR-EyeTracking `016.mp4` uses 3 reference frames where its own header allows 2; ffmpeg decodes it with a warning, Chrome stops with a decoding error, so it cannot be opened in OverGaze. 9 VR-EyeTracking files named `.mp4` are Matroska files (the container WebM is based on) with H.264 video: 001, 012, 019, 026, 039, 042, 055, 059, 166; OverGaze plays them and measures their frame rate instead of reading it from the header. The datasets were not changed.
