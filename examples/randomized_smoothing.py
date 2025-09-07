import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set style - no grid, clean background
plt.style.use('default')
# Use sans-serif math (\mathsf) and a TrueType sans font to get mathsf-style math
plt.rcParams['mathtext.fontset'] = 'stixsans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 20
plt.rcParams['text.usetex'] = False  # use matplotlib mathtext
plt.rcParams['mathtext.default'] = 'regular'
# embed TrueType fonts in PDF to avoid Type3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Define the non-smooth function
def relu(x):
    return np.maximum(0, x)

# Define Heaviside step function (gradient of ReLU)
def heaviside(x):
    return np.where(x > 0, 1.0, 0.0)

# Define Dirac delta (Hessian of ReLU) - theoretical, not directly computable
def dirac_delta(x):
    # Dirac delta is infinite at 0, 0 elsewhere - we'll represent it conceptually
    return np.where(np.abs(x) < 1e-10, np.nan, 0.0)

# Define covariance matrix and its Cholesky factor
def setup_covariance(sigma=1.0):
    Sigma = np.array([[sigma**2]])
    C = np.linalg.cholesky(Sigma)
    return Sigma, C

# Analytical solutions
def smoothed_relu(x, sigma=1.0):
    z = x / sigma
    return x * norm.cdf(z) + sigma * norm.pdf(z)

def smoothed_relu_gradient(x, sigma=1.0):
    return norm.cdf(x / sigma)

def smoothed_relu_hessian(x, sigma=1.0):
    """Analytical Hessian: PDF of normal distribution (logistic-like)"""
    return norm.pdf(x / sigma) / sigma

# Monte Carlo estimators
def monte_carlo_smoothed_relu(x, C, n_samples=1000):
    """Zeroth-order: f_ρ(x) = E[f(x + Cε)]"""
    epsilons = np.random.normal(0, 1, n_samples)
    transformed_noise = C[0, 0] * epsilons
    return np.mean(relu(x + transformed_noise))

def monte_carlo_gradient_zeroth_order(x, C, n_samples=1000):
    """Zeroth-order: ∇f_ρ(x) = E[C⁻¹ε f(x + Cε)]"""
    epsilons = np.random.normal(0, 1, n_samples)
    C_inv = 1.0 / C[0, 0]
    return np.mean(C_inv * epsilons * relu(x + C[0, 0] * epsilons))

def monte_carlo_gradient_first_order(x, C, n_samples=1000):
    """First-order: ∇f_ρ(x) = E[∇f(x + Cε)]"""
    epsilons = np.random.normal(0, 1, n_samples)
    x_perturbed = x + C[0, 0] * epsilons
    # Use Heaviside (subgradient of ReLU)
    return np.mean(heaviside(x_perturbed))

def monte_carlo_hessian_zeroth_order(x, C, n_samples=1000):
    """Zeroth-order: ∇²f_ρ(x) = E[C⁻²(ε² - 1) f(x + Cε)]"""
    epsilons = np.random.normal(0, 1, n_samples)
    C_inv2 = 1.0 / (C[0, 0] ** 2)
    return np.mean(C_inv2 * (epsilons**2 - 1) * relu(x + C[0, 0] * epsilons))

def monte_carlo_hessian_first_order(x, C, n_samples=1000):
    """First-order: ∇²f_ρ(x) = E[∇²f(x + Cε)] - this will fail!"""
    epsilons = np.random.normal(0, 1, n_samples)
    x_perturbed = x + C[0, 0] * epsilons
    # Dirac delta is 0 almost everywhere, so this will give 0
    return np.mean(np.where(np.abs(x_perturbed) < 1e-10, np.nan, 0.0))

# Generate data
x_values = np.linspace(-3, 3, 1000)
sigma = 1.0
Sigma, C = setup_covariance(sigma)

# Compute values
y_original = relu(x_values)
y_smoothed = smoothed_relu(x_values, sigma)
y_gradient_analytical = smoothed_relu_gradient(x_values, sigma)
y_hessian_analytical = smoothed_relu_hessian(x_values, sigma)

# Monte Carlo estimates
n_mc_samples = 5000
y_mc_smoothed = np.array([monte_carlo_smoothed_relu(xi, C, n_mc_samples) for xi in x_values])
y_mc_gradient_zeroth = np.array([monte_carlo_gradient_zeroth_order(xi, C, n_mc_samples) for xi in x_values])
y_mc_gradient_first = np.array([monte_carlo_gradient_first_order(xi, C, n_mc_samples) for xi in x_values])
y_mc_hessian_zeroth = np.array([monte_carlo_hessian_zeroth_order(xi, C, n_mc_samples) for xi in x_values])
y_mc_hessian_first = np.array([monte_carlo_hessian_first_order(xi, C, n_mc_samples) for xi in x_values])

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Function
ax1.plot(x_values, y_original, 'k-', alpha=0.8, linewidth=1, label=r'$g(x) = \max(0, x)$')
ax1.plot(x_values, y_mc_smoothed, 'b-', linewidth=2, label=r'$\hat g(x)(x)$ - R')
ax1.plot(x_values, y_smoothed, 'r--', alpha=1, linewidth=2.5, label=r'$\hat g(x)(x)$ - A')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$g(x)$')
# ax1.set_title('Function Approximation using Randomized Smoothing')
legend1 = ax1.legend(fontsize=18, handlelength=1.0, handletextpad=0.3,
                     labelspacing=0.2, columnspacing=0.6,
                     borderpad=0.2, borderaxespad=0.3, markerscale=0.8,
                     frameon=False)
legend1.set_draggable(True)
# force a draw so autoscale updates and we capture the final y-limits
fig.canvas.draw()
ylims = ax1.get_ylim()

# Plot 2: Gradient
ax2.plot(x_values, np.where(x_values > 0, 1, 0), 'k-', alpha=0.8, linewidth=1, label=r'$\nabla g(x) (Heaviside)$')
ax2.plot(x_values, y_mc_gradient_zeroth, 'b-', alpha=1, linewidth=2, 
         label=r'$\nabla \hat g(x)$ - RZ')
ax2.plot(x_values, y_mc_gradient_first, 'g-', alpha=1, linewidth=2, 
         label=r'$\nabla \hat g(x)$ - RF')
ax2.plot(x_values, y_gradient_analytical, 'r--', alpha=1, linewidth=2.5, 
         label=r'$\nabla \hat g(x)$ - A')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$\nabla g(x)$')
# ax2.set_title('Gradient Estimation using Randomized Smoothing')
legend2 = ax2.legend(fontsize=18, handlelength=1.0, handletextpad=0.3,
                     labelspacing=0.2, columnspacing=0.6,
                     borderpad=0.2, borderaxespad=0.3, markerscale=0.8,
                     frameon=False)
legend2.set_draggable(True)
# apply same y-limits
# ax2.set_ylim(ylims)

# Plot 3: Hessian
# For Dirac delta, show an arrow at x=0. Use the lower y-limit from the first plot and a manual upper limit.
manual_upper = 0.8  # set desired upper y-limit for the third subplot (change as needed)

# Clip Monte Carlo hessian results to the desired plotting range so autoscale won't expand limits
# (this forces the visual range while preserving the shape within bounds)
y_mc_hessian_zeroth_clipped = np.clip(y_mc_hessian_zeroth, ylims[0], manual_upper)
y_mc_hessian_first_clipped = np.clip(y_mc_hessian_first, ylims[0], manual_upper)
y_hessian_analytical_clipped = np.clip(y_hessian_analytical, ylims[0], manual_upper)

# position arrow near top of the specified range
arrow_y_top = 0.2
arrow_y_bottom = 0.0
ax3.annotate('', xy=(0, arrow_y_top), xytext=(0, arrow_y_bottom), 
             arrowprops=dict(arrowstyle='->', lw=1, color='black'),
             annotation_clip=False)
# add a proxy line so the legend shows the Dirac arrow entry
ax3.plot([], [], 'k-', lw=1, label=r'$\nabla^2 g(x)$ (Dirac Delta)')
# plot clipped data so autoscale respects the intended limits
ax3.plot(x_values, y_mc_hessian_zeroth_clipped, 'b-', alpha=1, linewidth=2, 
         label=r'$\nabla^2 \hat g(x)$ - RZ')
ax3.plot(x_values, y_mc_hessian_first_clipped, 'g-', alpha=1, linewidth=2, 
         label=r'$\nabla^2 \hat g(x)$ - RF')
ax3.plot(x_values, y_hessian_analytical_clipped, 'r--', alpha=1, linewidth=2.5, 
         label=r'$\nabla^2 \hat g(x)$ - A')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$\nabla^2 g(x)$')
# ax3.set_title('Hessian Estimation using Randomized Smoothing')
# enforce final limits and lock autoscaling
ax3.set_ylim(bottom=ylims[0], top=manual_upper)
ax3.set_autoscaley_on(False)
legend3 = ax3.legend(fontsize=18, handlelength=1.0, handletextpad=0.3,
                     labelspacing=0.2, columnspacing=0.6,
                     borderpad=0.2, borderaxespad=0.3, markerscale=0.8,
                     frameon=False)
legend3.set_draggable(True)

# Allow user to move legends interactively and capture final bbox on mouse release
def _on_legend_release(event):
    # write legend bbox positions (axes coords) to a file so user can reuse them
    try:
        with open('legend_positions.txt', 'a') as f:
            for idx, (ax, legend) in enumerate(((ax1, legend1), (ax2, legend2), (ax3, legend3)), start=1):
                bbox = legend.get_window_extent()
                bbox_axes = ax.transAxes.inverted().transform_bbox(bbox)
                vals = bbox_axes.bounds  # (x0, y0, width, height) in axes coords
                f.write(f'legend{idx}: {vals}\n')
    except Exception:
        pass

fig.canvas.mpl_connect('button_release_event', _on_legend_release)

# Clean up the appearance
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='both', which='both', length=4)
    ax.set_facecolor('white')

plt.tight_layout()
# draw and save
fig.canvas.draw()
plt.savefig('randomized_smoothing_with_hessian.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('randomized_smoothing_with_hessian.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
