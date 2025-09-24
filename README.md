Trajectory Optimization for non-smooth systems in robotics, using the idea of (randomized) smoothing from non-smooth optimization.

To get started, create environment using:
```
conda env create -f environment.yml
```
and run the examples from ./examples

## Simulation Results

Below are results on some simple use-cases:

- 1D Box Push — 1D box constrained to move along a single axis with dry Coulomb friction

  ![1D Box Push: 1D box constrained to move along single axis with dry Coulomb friction](results/animations/block_compare_animation.gif)

- Pendulum Swing Up — Pendulum with joint friction

  ![Pendulum Swing Up: Pendulum with joint friction](results/animations/pendulum_compare_animation.gif)

- Cartpole Swing Up — Cartpole with joint friction at the pole–cart joint

  ![Cartpole Swing Up: Cartpole with joint friction at pole-cart joint](results/animations/cartpole_compare_animation.gif)

