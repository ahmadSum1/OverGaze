# Both-eye version of the bundled 360° sample, in the Salient360 head + eye format (one file per eye, the index
# restarting at 0 for each recording, positions 0–1 from the top left, frames at 30 per second):
#   sample/eyes/L/sample_fixations.csv and sample/eyes/R/sample_fixations.csv
# The 8 simulated observers of sample/sample_gaze.csv are the people; the right eye's fixations start up to 20 ms
# apart and land up to about 1° away from the left eye's, and the right-eye file lists the people in another order
# (as in Salient360, where the two files do not line up). One extra recording in each file has no partner.
import csv, random
FPS = 30
rng = random.Random(11)
rows = list(csv.DictReader(open('sample/sample_gaze.csv')))
people = {}
for r in rows:
    people.setdefault(int(r['user']), []).append((float(r['start_time']), float(r['duration']), float(r['longitude']), float(r['latitude'])))
def eye(fix, dt_ms=0.0, jitter=0.0):
    out = []
    for t, d, lon, lat in fix:
        t2 = max(0.0, t + rng.uniform(-dt_ms, dt_ms) / 1000)
        lon2 = lon + rng.gauss(0, jitter); lat2 = lat + rng.gauss(0, jitter)
        out.append((t2, d + rng.uniform(-0.01, 0.01), lon2, lat2))
    return out
def write(path, recordings):
    with open(path, 'w') as f:
        f.write('# Idx, longitude, latitude, start timestamp, duration, start frame, end frame\n')
        for rec in recordings:
            for i, (t, d, lon, lat) in enumerate(rec):
                x = ((lon + 180) % 360) / 360; y = 0.5 - lat / 180
                f.write(f'{i},{x:.6f},{y:.6f},{t * 1000:.3f},{d * 1000:.3f},{int(t * FPS) + 1},{int((t + d) * FPS) + 1}\n')
left = [eye(people[p]) for p in sorted(people)]
right = [eye(people[p], 20, 0.6) for p in sorted(people)]
order = list(range(len(right))); rng.shuffle(order); right = [right[i] for i in order]
# one recording without a partner in each file: someone looking at other things, at other moments
def stranger(seed_shift):
    out, t = [], 0.2 + seed_shift
    while t < 7.8:
        d = rng.uniform(0.2, 0.4); out.append((t, d, rng.uniform(-170, 170), rng.uniform(-40, 40))); t += d + rng.uniform(0.05, 0.1)
    return out
left.append(stranger(0.0)); right.insert(3, stranger(0.13))
write('sample/eyes/L/sample_fixations.csv', left); write('sample/eyes/R/sample_fixations.csv', right)
print('right-eye file order of the people:', [order[i] + 1 for i in range(len(order))])
