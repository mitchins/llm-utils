import os
import sys
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Directory containing the TensorBoard event file(s)
log_dir = sys.argv[1] if len(sys.argv) > 1 else "./logs"


# Dynamically collect all class_eval F1 tags
ea = EventAccumulator(event_file)
ea.Reload()
all_tags = ea.Tags().get("scalars", [])
f1_tags = [tag for tag in all_tags if tag.startswith("class_eval/f1_")]

# Walk log_dir and locate event files
event_files = []
for root, _, files in os.walk(log_dir):
    for f in files:
        if f.startswith("events.out.tfevents"):
            event_files.append(os.path.join(root, f))

if not event_files:
    print("❌ No TensorBoard event files found.")
    sys.exit(1)

# Use the latest event file
event_file = sorted(event_files)[-1]
print(f"📊 Loading: {event_file}")

# Plotting
plt.figure(figsize=(12, 6))
for tag in f1_tags:
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, label=tag.split("/")[-1])
    except KeyError:
        print(f"⚠️  Tag not found: {tag}")

plt.title("F1 Score per Tone Class Over Steps")
plt.xlabel("Training Step")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
