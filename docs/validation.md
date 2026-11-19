# Check against the dataset maps

Settings in OverGaze: σ 3°, window ±1 s, Gaussian weights (σ = 0.5 s), speed filter 75°/s (VR-EyeTracking), motion analysed at 256 × 128 with the "play" method, motion compensation on. The app's raw maps (512 × 256) were averaged to 64 × 128 blocks, as in the reference maps (`maps_mc3_*`, motion-compensated pooling by the data agent), and compared on 40–117 frames per clip. Coverage = share of the frame at or above 0.1 of the frame maximum, per pixel, mean over the sampled frames. CC = correlation of app map and reference map.

| clip | coverage app | coverage reference | CC with reference |
|---|---|---|---|
| VR-EyeTracking 001 | 6.20 % | 6.33 % | 0.985 |
| VR-EyeTracking 002 | 2.23 % | 2.29 % | 0.947 |
| VR-EyeTracking 092 | 2.62 % | 2.68 % | 0.994 |
| VR-EyeTracking 117 | 1.69 % | 1.43 % | 0.937 |
| VR-EyeTracking 050 (60 fps, reference uses every 2nd frame) | 6.65 % | 6.43 % | 0.950 |
| YT360 `-ZrvrKMC7Tc-28` (6 conditions pooled, 29.97 fps) | 7.47 % | 7.46 % | 0.945 |

Over all 132 reference VR-EyeTracking clips the mean coverage is 3.43 % (per pixel); the five clips above average 3.8 % in both app and reference. The YT360 correlation is highest at lag 0 (0.945; 0.87 at ±10 frames), so time alignment, orientation and the clip-start offset are right. A 25 fps YT360 clip (`pyRnPVbCmII-543`) is measured as 25 fps.

Not committed with images: the clips have no licence for redistribution, so the screenshots in `docs/screenshots` use the synthetic samples.
