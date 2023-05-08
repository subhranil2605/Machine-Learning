import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Styling the matplotlib
plt.style.use("seaborn-v0_8-dark")

for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey

colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41',  # matrix green
]


def forward_loss(X_batch: ndarray,
                 y_batch: ndarray,
                 weights: ndarray):
    assert X_batch.shape[0] == y_batch.shape[0]
    assert X_batch.shape[1] == weights.shape[0]

    # performing dot product
    h: ndarray = X_batch.dot(weights)

    # calculating the cost function
    J: float = np.mean(np.power(y_batch - h, 2))

    forward_info: dict[str, ndarray] = {
        "X": X_batch,
        "h": h,
        "y": y_batch
    }

    return J, forward_info


def loss_gradients(forward_info: dict[str, ndarray]) -> ndarray:
    dLdh: ndarray = (2 / len(forward_info["X"])) * \
        (forward_info["h"] - forward_info["y"])
    dhdt: ndarray = np.transpose(forward_info["X"], (1, 0))
    dLdt: ndarray = np.dot(dhdt, dLdh)
    return dLdt


def init_weights(X: ndarray) -> ndarray:
    theta: ndarray = np.zeros(X.shape[1])
    return theta


def train(X: ndarray,
          y: ndarray,
          learning_rate: float = 0.01,
          n_iter: int = 1000,
          fit_intercept: bool = True):

    if fit_intercept:
        X: ndarray = np.vstack([np.ones(len(X)), X]).T
    else:
        X: ndarray = X.reshape(-1, 1)

    # initialize weights
    theta: ndarray = init_weights(X)

    losses = []
    thetas = []

    for i in range(n_iter):
        loss, foward_info = forward_loss(X, y, theta)

        losses.append(loss)

        loss_grads = loss_gradients(foward_info)

        theta -= learning_rate * loss_grads
        thetas.append(theta.tolist()[:])

    return losses, thetas


# number of training examples
n = 1000


def get_y(x: ndarray, theta: ndarray):
    X = np.vstack([np.ones(x.shape[0]), x]).T
    return X.dot(theta)


class UpdateLine:
    def __init__(self, ax):
        self.line, = ax[0].plot([], [], color=colors[0])
        self.loss_line, = ax[1].plot([], [], color=colors[1])
        self.x: ndarray = np.linspace(0, 10, 100)
        self.y: ndarray = 3 * self.x + 10 + np.random.randn(100) * 2
        self.losses, self.thetas = train(self.x, self.y, n_iter=n)
        self.ax_0 = ax[0]
        self.ax_1 = ax[1]

        # Set up plot parameters
        self.ax_0.set_xlim(self.x.min(), self.x.max())
        self.ax_0.set_ylim(self.y.min(), self.y.max())
        self.ax_0.scatter(self.x, self.y, color=colors[2])
        self.text = ax[0].text(0.05, 0.85, '', transform=ax[0].transAxes)
        self.loss_text = ax[1].text(0.20, 0.90, '', transform=ax[1].transAxes)

        self.ax_1.set_xlim(-5, 200)
        self.ax_1.set_ylim(np.array(self.losses).min() - 5,
                           np.array(self.losses).max() + 5)

        self.xdata = []
        self.ydata = []

    def __call__(self, i):
        if i == 0:
            self.line.set_data([], [])
            self.loss_line.set_data([], [])
            return self.line, self.loss_line

        y = get_y(self.x, self.thetas[i])
        self.line.set_data(self.x, y)
        self.text.set_text(
            f"n_iter={i}\nintercept={self.thetas[i][0]}\nweight={self.thetas[i][1]}")

        xmin, xmax = ax[1].get_xlim()

        self.xdata.append(i - 1)
        self.ydata.append(self.losses[i - 1])

        if i >= xmax:
            ax[1].set_xlim(xmin, 2*xmax)
            ax[1].figure.canvas.draw()
        self.loss_line.set_data(self.xdata, self.ydata)
        self.loss_text.set_text(f"loss={self.losses[i]}")

        return self.line, self.text, self.loss_line, self.loss_text



fig, ax = plt.subplots(1, 2, figsize=(16, 9))
ax[0].grid(color='#2A3459')
ud = UpdateLine(ax)
anim = animation.FuncAnimation(
    fig, ud, frames=n, interval=10, blit=True, repeat=False)

fig.canvas.manager.full_screen_toggle()

# Save the animation as an mp4 video
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# fig.set_size_inches(16, 9)  # set the aspect ratio to 16:9
# anim.save('animation.mp4', writer=writer, dpi=150)


plt.show()
