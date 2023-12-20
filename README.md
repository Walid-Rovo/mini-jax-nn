# mini-jax-nn
Minimal neural network demonstration built using JAX. This serves as a starting point to creating
your own neural network. The network is trained using
[Mini-batch SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method) and is
taught to predict the forward integration step of the Lorenz Attractor ODE.

![Lorenz attractor animation](./lorenz_animation.webp)

# Requirements
You need a Python interpreter with JAX and matplotlib. Example:
```sh
python3 -m venv .venv  # Create a new virtual environment in the '.venv' directory
source .venv/bin/activate  # Activate the virtual environment
pip install jax jaxlib matplotlib  # Install packages using pip
```

# Tips for Further Development
A logical next step would be to make the Python class more portable by working around JAX's
limitations using PyTrees:
https://jax.readthedocs.io/en/latest/pytrees.html

One could also implement more
[adaptive](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad) Gradient Descent
methods like AdaGrad, RMSProp and Adam.

# Citations
JAX:
```bibtex
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}
```
JAX tutorials, specifically: https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html