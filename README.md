# Mpemba Effect in a Two-Dimensional Bistable Potential

This repository contains the Python codes used to generate the figures presented in

H. Hayakawa and S. Takada,  
**"Mpemba effect in a two-dimensional bistable potential"**  
[arXiv:2603.24148](https://doi.org/10.48550/arXiv.2603.24148)

## Usage

The codes for the figures are essentially independent and can be executed separately.

For most figures, running the corresponding Python script directly generates the figure.

For **Fig. 10**, the calculation and plotting are separated into two steps:

1. Run `Fig10_step1.py` to calculate the phase-diagram data and save them to a `.dat` file.
2. Run `Fig10_step2.py` to read the generated data file and plot the phase diagram.

The same procedure is used for **Fig. 13**:

1. Run `Fig13_step1.py` to generate the data file.
2. Run `Fig13_step2.py` to plot the phase diagram from the generated data.

## Requirements

The codes require Python 3 and the following packages:

- NumPy
- SciPy
- Matplotlib

They can be installed using

```bash
pip install numpy scipy matplotlib
