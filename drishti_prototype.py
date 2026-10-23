"""
DRISHTI - prototype (simulation only)
Multi-sensor fusion + V2X early warning for mine trucks in fog.

No hardware needed: sensors are simulated, so we can show the logic
(voting, Kalman filter, V2X, speed governor) on a laptop.
Run:  python drishti_prototype.py
"""
import math
import random
import time

random.seed(7)

# ---------- simulated sensors ----------
# each one returns distance to obstacle in metres, or None if it sees nothing

def thermal_cam(true_dist, visibility):
    # thermal works ok in fog but gets worse in very dense fog
    if true_dist > 60 + visibility * 0.4:
        return None
    return true_dist + random.gauss(0, 2 + (100 - visibility) * 0.05)

def radar(true_dist, visibility):
    # 77GHz radar barely cares about fog
    if true_dist > 150:
        return None
    return true_dist + random.gauss(0, 1)

def ultrasonic(true_dist, visibility):
    # short range only, but good for close calls
    if true_dist > 8:
        return None
    return true_dist + random.gauss(0, 0.3)

def v2x_distance(my_pos, other_pos):
    # other vehicle broadcasts GPS over LoRa/5G, we compute the gap
    gap = other_pos - my_pos
    return gap + random.gauss(0, 1.5) if 0 < gap < 250 else None


# ---------- simple 1D Kalman filter for the fused distance ----------
class DistanceFilter:
    def __init__(self):
        self.x = None      # estimated distance
        self.p = 100.0     # uncertainty

    def update(self, measurements, closing_speed, dt):
        # predict: gap shrinks by closing_speed * dt
        if self.x is not None:
            self.x -= closing_speed * dt
            self.p += 1.0
        # update with each sensor reading
        for z, r in measurements:
            if self.x is None:
                self.x, self.p = z, r
                continue
            k = self.p / (self.p + r)
            self.x += k * (z - self.x)
            self.p *= (1 - k)
        return self.x


# ---------- decision logic ----------
def safe_speed(distance):
    """Max speed (km/h) so truck can stop before the obstacle.
    Rough braking model: v = sqrt(2 * a * (d - margin))"""
    a = 2.5          # braking decel in m/s^2, loaded dumper
    margin = 10
    if distance <= margin:
        return 0
    return min(40, math.sqrt(2 * a * (distance - margin)) * 3.6)


def slow(text, delay=0.6):
    print(text, flush=True)
    time.sleep(delay)


def boot():
    print("=" * 70)
    slow("  D R I S H T I  -  Fog-Safe Mine Haulage System", 1)
    print("=" * 70)
    slow("Starting edge unit...", 1)
    for name in ["Thermal camera", "77GHz mmWave radar", "Ultrasonic sensor",
                 "GNSS / RTK GPS", "V2X radio (LoRa/5G)"]:
        print(f"Scanning {name:<22}", end="", flush=True)
        for _ in range(3):
            time.sleep(0.4)
            print(".", end="", flush=True)
        print(" ONLINE")
    slow("\nVisibility check: DENSE FOG detected (about 20%)", 1)
    slow("Switching to multi-sensor fusion mode", 1)
    slow("Truck moving. Watching the road ahead...\n", 1)


def run():
    my_pos, my_speed = 0.0, 30 / 3.6       # truck starts at 30 km/h
    stopped_vehicle = 220.0                 # parked dozer ahead, hidden by fog
    visibility = 20                         # 0-100, low = thick fog
    dt = 0.5
    kf = DistanceFilter()
    known = set()
    boot()

    print(f"{'t(s)':>5} {'true':>6} {'fused':>6} {'sensors':<22} {'speed':>6}  status")
    print("-" * 70)

    for step in range(120):
        t = step * dt
        true_d = stopped_vehicle - my_pos

        readings = {
            "thermal": thermal_cam(true_d, visibility),
            "radar": radar(true_d, visibility),
            "ultra": ultrasonic(true_d, visibility),
            "v2x": v2x_distance(my_pos, stopped_vehicle),
        }
        seen = {k: v for k, v in readings.items() if v is not None}

        # noise levels per sensor (variance) - used as Kalman weights
        noise = {"thermal": 9, "radar": 1.5, "ultra": 0.2, "v2x": 4}
        fused = kf.update([(v, noise[k]) for k, v in seen.items()],
                          my_speed, dt)

        # voting: only raise alarm if 2+ sensors agree (cuts false alarms)
        votes = len(seen)
        status = "clear"
        target = 40.0
        if fused is not None and votes >= 2:
            target = safe_speed(fused)
            if fused < 40:
                status = "BRAKE!"
            elif fused < 100:
                status = "WARNING"
            else:
                status = "caution"
        elif votes == 1:
            status = "1 sensor only (ignored)"

        # governor eases the truck toward the allowed speed
        cur = my_speed * 3.6
        if cur > target:
            cur = max(target, cur - 12 * dt)     # auto-brake
        elif cur < target and status == "clear":
            cur = min(target, cur + 3 * dt)
        my_speed = cur / 3.6
        my_pos += my_speed * dt

        for k in seen:
            if k not in known:
                known.add(k)
                print(f"  >>> {k.upper()} has detected an object at {seen[k]:.0f} m!")
                time.sleep(0.8)

        names = ",".join(seen) or "-"
        f = f"{fused:6.1f}" if fused is not None else "     -"
        print(f"{t:5.1f} {true_d:6.1f} {f} {names:<22} {cur:6.1f}  {status}")
        time.sleep(0.25)

        if my_speed < 0.1 and true_d < 30:
            print("\n*** AUTO-BRAKE ENGAGED ***")
            time.sleep(1)
            print(f"Truck stopped safely {true_d:.1f} m from the obstacle.")
            break
    else:
        print("\nSimulation ended without reaching the obstacle.")


if __name__ == "__main__":
    run()
    input("\nPress Enter to exit...")
