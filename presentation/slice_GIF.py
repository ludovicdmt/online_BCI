import imageio
import numpy as np

# Open the GIF file
gif = imageio.mimread("presentation\\img\\leg.gif")

# Extract individual frames
frames = []
for i, frame in enumerate(gif):
    if 20 <= i <= 37:
        frame_name = f"presentation\\img\\leg\\frame_{i-20}.png"
        # Flip the image vertically
        # flipped_frame = frame[:, ::-1, :]
        imageio.imwrite(frame_name, frame)
        # frame_name = f'presentation\\img\\biceps_right\\frame_{39-i}.png'
        # imageio.imwrite(frame_name, flipped_frame)
        frames.append(frame_name)


print(f"Extracted {len(frames)} frames from the GIF.")
