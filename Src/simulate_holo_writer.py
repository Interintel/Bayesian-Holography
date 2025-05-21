import numpy as np
import time
import os
import matplotlib.pyplot as plt
import random

FILENAME = "holo.npy"
HEIGHT, WIDTH = 64, 64
MODE = "mixed"  # Options: "motion", "random", "mixed"
INTERVAL = 0.5  # Seconds between updates
MOTION_SPEED = 1

def generate_motion_frame(frame, step):
    return np.roll(frame, step, axis=1)

def generate_random_frame():
    return np.random.rand(HEIGHT, WIDTH)

def blend_frames(frame1, frame2, alpha=0.5):
    return (1 - alpha) * frame1 + alpha * frame2

def simulate():
    MODE = "motion"
    print(f"🌀 Starting simulation: mode={MODE}, output={FILENAME}")
    base_frame = np.zeros((HEIGHT, WIDTH))
    base_frame[20:30, 10:20] = 1.0  # Simple blob for motion

    motion_step = 0

    plt.ion()
    fig, ax = plt.subplots()
    heatmap = ax.imshow(base_frame, cmap='hot', interpolation='nearest')
    cbar = plt.colorbar(heatmap)
    ax.set_title("Simulated Hologram Frame")
    
    while True:
        random_mode = random.randint(1, 3)
        if random_mode == 1:
            MODE = "motion"
        elif random_mode == 2:
            MODE = "random"
        elif random_mode == 3:
            MODE = "mixed"

        if MODE == "motion":
            frame = generate_motion_frame(base_frame, motion_step)
            motion_step = (motion_step + MOTION_SPEED) % WIDTH
        elif MODE == "random":
            frame = generate_random_frame()
        elif MODE == "mixed":
            motion = generate_motion_frame(base_frame, motion_step)
            noise = generate_random_frame()
            frame = blend_frames(motion, noise, alpha=0.3)
            motion_step = (motion_step + MOTION_SPEED) % WIDTH
        else:
            raise ValueError("Unknown MODE")

        np.save(FILENAME, frame)
        print(f"💾 Frame written at {time.strftime('%H:%M:%S')} Mode:", MODE)

        # Update heatmap
        heatmap.set_data(frame)
        ax.set_title(f"Simulated Frame @ {time.strftime('%H:%M:%S')}")
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(INTERVAL)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(FILENAME) or ".", exist_ok=True)
    simulate()
