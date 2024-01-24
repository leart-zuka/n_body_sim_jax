# N Body Simulation 

This is an N body simulation coded in Python with the help of a package called JAX,
which can be used to accelerate matrix matrix multiplications or numerical calculations by using GPUs.
___
# Requirements

It is required to have the following packages installed

- numpy
- matplotlib
- PyQt6 (an interactive matplotlib backend)
- jaxlib
- jax (if you want to use the requirements.txt in this project it will download jax for GPUs that can use CUDA 12)

___
# Install

If you want to install the packages just run:
```shell
pip install -r requirements
```
___
# Run

Running the project is as simple as running a normal python file, so just:
```shell
python main.py
```

It should be noted that depending on your GPU you might need to adjust the amount of particles,or chunk size, or the dtype of the matrices,
as more particles and a larger chunk size can take up more VRAM leading to the program not running because of a lack of memory
