#!/usr/bin/env python3
"""Motion file for OverGaze made with OpenCV's dense optical flow (DIS, "dense inverse search").

Load the result with "Load motion file" in OverGaze's motion compensation section, instead of "Analyse motion".
The frames are sampled the way OverGaze's "Analyse motion" (seek mode) samples them: from the start, every
1 / rate seconds, rounded to the frame that contains that time; each motion field goes from one sampled frame to the
next. Frames are made greyscale and shrunk to width × width/2 (360° video) before the flow is computed; for 360°
video the left and right edges are joined (the picture is padded with its other side first).

    python tools/motion_opencv.py video.mp4                       # → video.motion.npz, 256 × 128, 10 per second
    python tools/motion_opencv.py video.mp4 --width 256 --rate 10 --start 0 --end 20 --flat   # ordinary 2D video

Needs numpy and opencv-python. The file is a NumPy .npz archive:
    t_a, t_b   (N,) float64   times (s) of the two frames of each field (frame centres)
    flow       (N, H, W, 2)   float16, movement in pixels at H × W from the frame at t_a to the frame at t_b
                              (x to the right, y down): the content at (x, y) in the first frame is at
                              (x + flow[..., 0], y + flow[..., 1]) in the second
    wrap       ()  bool       True when the left and right edges meet (360° video)
    fps        ()  float64    the video's frame rate as OpenCV read it
    method     ()  str        how the flow was computed
"""
import argparse, math, sys
import numpy as np
import cv2


def sample_frames(cap_fps, n_frames, start, end, rate):
    """Frame indexes OverGaze's seek mode grabs: t = start + i / rate, frame k = floor(t · fps)."""
    ks, t = [], start
    i = 0
    while True:
        t = start + i / rate
        if t > end - 0.25 / cap_fps:
            break
        k = max(0, math.floor(t * cap_fps + 1e-6))
        if k < n_frames and (not ks or k != ks[-1]):
            ks.append(k)
        i += 1
    return ks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('-o', '--out', help='output .npz (default: <video>.motion.npz)')
    ap.add_argument('--width', type=int, default=256, help='flow width in pixels (default 256)')
    ap.add_argument('--height', type=int, help='flow height (default: width / 2 for 360° video, else from the aspect ratio)')
    ap.add_argument('--rate', type=float, default=10, help='motion fields per second (default 10)')
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--end', type=float, default=None)
    ap.add_argument('--fps', type=float, default=None, help='frame rate, if OpenCV reads it wrongly')
    ap.add_argument('--flat', action='store_true', help='ordinary 2D video: no joining of the left and right edges')
    ap.add_argument('--preset', default='medium', choices=['ultrafast', 'fast', 'medium'], help='DIS preset (default medium)')
    ap.add_argument('--save-frames', help='also save the greyscale frames used (uint8 .npy, N × H × W) here')
    a = ap.parse_args(argv)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f'cannot open {a.video}')
    fps = a.fps or cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wrap = not a.flat
    W = a.width
    H = a.height or (W // 2 if wrap else max(1, round(W * vh / vw)))
    end = a.end if a.end is not None else n / fps
    ks = sample_frames(fps, n, a.start, end, a.rate)
    want = set(ks)

    frames, k = {}, 0
    while k <= ks[-1]:
        ok = cap.grab()
        if not ok:
            break
        if k in want:
            ok, img = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames[k] = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        k += 1
    ks = [k for k in ks if k in frames]
    if len(ks) < 2:
        sys.exit('fewer than two frames could be read')

    dis = cv2.DISOpticalFlow_create({'ultrafast': 0, 'fast': 1, 'medium': 2}[a.preset])
    pad = W // 8 if wrap else 0
    flows = np.zeros((len(ks) - 1, H, W, 2), np.float16)
    for i in range(len(ks) - 1):
        A, B = frames[ks[i]], frames[ks[i + 1]]
        if pad:
            A = np.concatenate([A[:, -pad:], A, A[:, :pad]], axis=1)
            B = np.concatenate([B[:, -pad:], B, B[:, :pad]], axis=1)
        f = dis.calc(A, B, None)
        flows[i] = f[:, pad:pad + W] if pad else f
        print(f'\r{i + 1} / {len(ks) - 1} fields', end='', file=sys.stderr)
    print(file=sys.stderr)

    t = (np.array(ks, np.float64) + 0.5) / fps
    out = a.out or a.video.rsplit('.', 1)[0] + '.motion.npz'
    np.savez_compressed(out, t_a=t[:-1], t_b=t[1:], flow=flows, wrap=np.array(wrap), fps=np.array(fps, np.float64),
                        method=np.array(f'OpenCV {cv2.__version__} DIS optical flow, preset {a.preset}, greyscale {W} x {H}'
                                        + (', left and right edges joined' if wrap else '')))
    if a.save_frames:
        np.save(a.save_frames, np.stack([frames[k] for k in ks]))
    print(f'{out}: {len(ks) - 1} fields at {W} x {H}, {t[0]:.3f}-{t[-1]:.3f} s, frame rate {fps:.5f}')


if __name__ == '__main__':
    main()
