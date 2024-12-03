import matplotlib.pyplot as plt

# Read the float values from the text file
with open('/home/anthony/HACO/haco/run_baselines/log2.txt', 'r') as file:
    losses = [float(line.strip()) for line in file]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Save the plot as an image file
plt.savefig('loss_graph.png')

# Display the plot (optional)
plt.show()
