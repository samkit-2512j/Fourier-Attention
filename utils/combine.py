import matplotlib.pyplot as plt
import re

# Define the paths for all log files
data_path_phase1 = "./phase1.log"
data_path_phase2 = "./phase2.log"
data_path_base = "./base.log"  # Path for the base log file

# Function to extract loss and epoch from a log file
def extract_loss_and_epoch(data_path):
    with open(data_path, 'r') as file:
        log_data = file.read()

    # Regular expressions for extracting 'loss' and 'epoch'
    loss_pattern = r"(?<![a-zA-Z_])'loss':\s([0-9.]+)"
    epoch_pattern = r"'epoch':\s([0-9.]+)"

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
    
    return epochs, losses

# Extract loss and epoch values from all three log files
epochs_phase1, losses_phase1 = extract_loss_and_epoch(data_path_phase1)
epochs_phase2, losses_phase2 = extract_loss_and_epoch(data_path_phase2)
epochs_base, losses_base = extract_loss_and_epoch(data_path_base)

# Add 5 to the epochs from the Phase 2 log (starting from the last epoch of Phase 1)
epochs_phase2 = [epoch + 5 for epoch in epochs_phase2]

# Print a few values to verify
print("Sample losses from Phase 1:", losses_phase1[:20])
print("Sample epochs from Phase 1:", epochs_phase1[:20])

print("Sample losses from Phase 2:", losses_phase2[:20])
print("Sample epochs from Phase 2:", epochs_phase2[:20])

print("Sample losses from Base log:", losses_base[:20])
print("Sample epochs from Base log:", epochs_base[:20])

# Plotting

plt.style.use('dark_background')  # Use a dark background for the plot

# Plot Phase 1 and Phase 2 (with epoch shift) together on the same plot
plt.plot(epochs_phase1, losses_phase1, marker='o', label='Phase 1 Log', color='cyan')
plt.plot(epochs_phase2, losses_phase2, marker='x', label='Phase 2 Log (Epoch + 5)', color='magenta')

# Plot the Base log separately
plt.plot(epochs_base, losses_base, marker='s', label='Base Log', color='yellow')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch (Phase 1, Phase 2 with Epoch Shift, and Base Log)')
plt.grid(True)

# Add a legend to differentiate the three logs
plt.legend()

# Show the plot
plt.show()
