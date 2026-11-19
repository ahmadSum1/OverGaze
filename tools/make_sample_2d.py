# Plain 2D sample: a red square moves over a noisy background; 6 simulated observers fixate it (pixel coordinates).
# Needs ffmpeg. Output: sample/sample_2d.mp4, sample/sample_2d_gaze.csv
import math, random, subprocess, os
W, H, FPS, DUR = 640, 360, 30, 6
random.seed(3)
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sample')
bx = lambda t: 60 + 80 * t          # box top-left, px
by = lambda t: 150 + 60 * math.sin(2 * t)
subprocess.run(['ffmpeg', '-y', '-v', 'error',
                '-f', 'lavfi', '-i', f'color=c=0x2a3a4a:s={W}x{H}:r={FPS}:d={DUR},noise=alls=8:allf=t',
                '-f', 'lavfi', '-i', f'color=c=0xe03030:s=40x40:r={FPS}:d={DUR}',
                '-filter_complex', "[0][1]overlay=x='60+80*t':y='150+60*sin(2*t)'",
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '33', '-movflags', '+faststart',
                os.path.join(here, 'sample_2d.mp4')], check=True)
rows = ['user,start_time,duration,x_px,y_px']
for u in range(6):
    t = 0.1 + 0.05 * u
    while t < DUR - 0.3:
        d = random.uniform(0.2, 0.45)
        tc = t + d / 2
        rows.append(f'{u},{t:.3f},{d:.3f},{bx(tc) + 20 + random.gauss(0, 6):.1f},{by(tc) + 20 + random.gauss(0, 6):.1f}')
        t += d + random.uniform(0.03, 0.08)
open(os.path.join(here, 'sample_2d_gaze.csv'), 'w').write('\n'.join(rows) + '\n')
