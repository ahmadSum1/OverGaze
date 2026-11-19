# Synthetic 360° sample: panning camera (scene moves left at PAN deg/s), a red ball moving on its own,
# a yellow sign fixed in the scene. 8 simulated observers fixate mostly the ball, sometimes the sign.
import numpy as np, subprocess, sys, csv
W, H, FPS, DUR = 768, 384, 30, 8.0
PAN = -20.0                      # scene (background) image motion, deg/s
rng = np.random.default_rng(7)
# world panorama
world = np.zeros((H, W, 3), np.float32)
yy = np.linspace(0, 1, H)[:, None]
world[:] = (0.25 + 0.35 * (1 - yy))[..., None] * np.array([0.55, 0.7, 1.0])[None, None]
world[H // 2:] = (0.18 + 0.25 * yy[H // 2:])[..., None] * np.array([0.55, 0.5, 0.35])[None, None]
for _ in range(90):  # textured "buildings" / rocks
    x0, w = rng.integers(0, W), rng.integers(8, 40)
    top, h = rng.integers(int(H * 0.3), int(H * 0.55)), rng.integers(10, 70)
    col = rng.uniform(0.15, 0.85, 3)
    xs = np.arange(x0, x0 + w) % W
    world[top:top + h, xs] = col * (0.8 + 0.2 * rng.random((h, len(xs), 1)))
world += 0.04 * rng.standard_normal(world.shape).astype(np.float32)
SIGN_LON, SIGN_LAT = 100.0, 8.0  # in world coords at t = 0
def lon2x(lon): return (lon / 360 + 0.5) * W - 0.5
def lat2y(lat): return (0.5 - lat / 180) * H - 0.5
def ball(t): return (-70 + 18 * t + 0.0, 5 + 12 * np.sin(2 * np.pi * t / 4))   # image lon/lat, deg
def sign(t):
    lon = SIGN_LON + PAN * t
    return (((lon + 180) % 360) - 180, SIGN_LAT)
X, Y = np.meshgrid(np.arange(W), np.arange(H))
def disk(img, lon, lat, r_deg, color):
    cx, cy = lon2x(lon), lat2y(lat)
    dx = ((X - cx + W / 2) % W) - W / 2
    r = r_deg / 360 * W
    d = np.sqrt(dx ** 2 + (Y - cy) ** 2)
    a = np.clip(r - d + 0.5, 0, 1)[..., None]
    img[:] = img * (1 - a) + a * np.array(color, np.float32)
n = int(DUR * FPS)
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-preset', 'slow', '-crf', '27', '-pix_fmt', 'yuv420p', '-g', '30', '-movflags', '+faststart', sys.argv[1]], stdin=subprocess.PIPE)
for k in range(n):
    t = k / FPS
    shift = int(round(PAN * t / 360 * W))
    img = np.roll(world, shift, axis=1).copy()
    sl, sa = sign(t)
    disk(img, sl, sa, 5, (0.98, 0.85, 0.15))
    bl, ba = ball(t)
    disk(img, bl, ba, 6.5, (0.02, 0.02, 0.02))
    disk(img, bl, ba, 5.5, (0.95, 0.12, 0.1))
    disk(img, bl + 1.5, ba + 1.5, 2.0, (1.0, 0.85, 0.8))
    ff.stdin.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
ff.stdin.close(); ff.wait()
rows = []
for obs in range(1, 9):
    t = rng.uniform(0.0, 0.3)
    while t < DUR - 0.1:
        dur = float(rng.uniform(0.18, 0.45))
        mid = min(DUR, t + dur / 2)
        target = sign(mid) if rng.random() < 0.18 else ball(mid)
        lon = target[0] + rng.normal(0, 1.2); lat = target[1] + rng.normal(0, 1.2)
        rows.append((obs, round(t, 3), round(dur, 3), round(((lon + 180) % 360) - 180, 3), round(lat, 3)))
        t += dur + float(rng.uniform(0.03, 0.08))
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['user', 'start_time', 'duration', 'longitude', 'latitude']); w.writerows(rows)
print(len(rows), 'fixations')
