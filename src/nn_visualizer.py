import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class NeuralNetworkVisualizer:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Node coordinates for layers
        self.node_pos = []
        self._compute_layout()

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.axis('off')

    def _compute_layout(self):
        """Compute (x, y) positions for all nodes."""
        self.node_pos = []
        # x positions for layers
        xs = [0, 1, 2]
        # y positions for each layer (centered)
        input_ys = np.linspace(0, 1, self.input_dim)
        hidden_ys = np.linspace(0, 1, self.hidden_dim)
        output_ys = np.linspace(0, 1, self.output_dim)
        # Store (x, y, layer_idx, node_idx)
        for i, y in enumerate(input_ys):
            self.node_pos.append((xs[0], y, 0, i))
        for i, y in enumerate(hidden_ys):
            self.node_pos.append((xs[1], y, 1, i))
        for i, y in enumerate(output_ys):
            self.node_pos.append((xs[2], y, 2, i))

    def draw(self, input_vals, hidden_vals, output_vals):
        """Draw the network, coloring nodes by activation."""
        self.ax.clear()
        self.ax.axis('off')

        # Prepare all activations
        activs = [
            np.array(input_vals).flatten(),
            np.array(hidden_vals).flatten(),
            np.array(output_vals).flatten()
        ]
        # Normalize activations for color scale
        all_activs = np.concatenate(activs)
        min_val, max_val = np.min(all_activs), np.max(all_activs)

        def get_color(val):
            norm = (val - min_val) / (max_val - min_val + 1e-6)
            return plt.cm.viridis(norm)

        # Draw edges: input->hidden and hidden->output
        for in_idx in range(self.input_dim):
            for hid_idx in range(self.hidden_dim):
                x0, y0 = 0, np.linspace(0, 1, self.input_dim)[in_idx]
                x1, y1 = 1, np.linspace(0, 1, self.hidden_dim)[hid_idx]
                self.ax.plot([x0, x1], [y0, y1], 'gray', lw=0.5)
        for hid_idx in range(self.hidden_dim):
            for out_idx in range(self.output_dim):
                x0, y0 = 1, np.linspace(0, 1, self.hidden_dim)[hid_idx]
                x1, y1 = 2, np.linspace(0, 1, self.output_dim)[out_idx]
                self.ax.plot([x0, x1], [y0, y1], 'gray', lw=0.5)

        # Draw nodes
        for (x, y, layer, idx) in self.node_pos:
            act = activs[layer][idx]
            circ = mpatches.Circle((x, y), 0.045, color=get_color(act), ec='k', lw=1.2)
            self.ax.add_patch(circ)

        # Annotate layers
        self.ax.text(0, 1.07, "Input Layer", ha="center", fontsize=12)
        self.ax.text(1, 1.07, "Hidden Layer", ha="center", fontsize=12)
        self.ax.text(2, 1.07, "Output Layer", ha="center", fontsize=12)
        plt.pause(0.001)
