import matplotlib.pyplot as plt
import re

data_path = "./output(3).log"

with open(data_path, 'r') as file:
    log_data = file.read()

# Precise regex for 'loss' and 'epoch' only, not 'eval_loss'
loss_pattern = r"(?<![a-zA-Z_])'loss':\s([0-9.]+)"
epoch_pattern = r"'epoch':\s([0-9.]+)"

# Lists to store the results
losses = []
epochs = []

# Loop through each match and ensure they belong to training loss
for line in log_data.splitlines():
    if "'loss':" in line and "'eval_loss':" not in line:  # Check if it's a training loss
        loss_match = re.search(loss_pattern, line)
        epoch_match = re.search(epoch_pattern, line)
        if loss_match and epoch_match:
            losses.append(float(loss_match.group(1)))
            epochs.append(float(epoch_match.group(1)))

# Print a few values to verify
print("Sample losses:", losses[:20])
print("Sample epochs:", epochs[:20])
print("Number of losses:", len(losses))
print("Number of epochs:", len(epochs))

# Plotting
plt.plot(epochs, losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.show()
