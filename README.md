# Graph SLAM Simulator

This is a repository for generating random single- and multi-robot graph SLAM experiments. The simulator makes the distinction between robot states and map features to enable *separability* in multi-robot SLAM system global cost functions, and logs the resulting graph to [PyFactorGraph](https://github.com/MarineRoboticsGroup/PyFactorGraph) format. The simulator focuses on the underlying graphical structure of the RA-SLAM problem, and assumes measurement models are consistent with those presented in the paper:

```bibtex
@article{papalia2023certifiably,
  title={Certifiably Correct Range-Aided SLAM},
  author={Papalia, Alan and Fishberg, Andrew and O'Neill, Brendan W. and How, Jonathan P. and Rosen, David M. and Leonard, John J.},
  journal={arXiv preprint arXiv:2302.11614},
  year={2023}
}
```

The simulator supports the generation of i) random Manhattan worlds and ii) controlled experiments with parameterized robot navigation and map feature placement. Supported measurement types include:
- Pose priors in SE(2) and SE(3)
- Relative SE(2) and SE(3) measurements (pose-pose and pose-point) for robot odometry and loop closures
- Range measurements in 2D and 3D for pose-pose, pose-point, and point-point pairs

## Getting Started

### Example

Run the example:

```bash
cd ~/manhattan-world-sim/examples
python3 example.py
```

### Dependencies

PyFactorGraph:
```bash
git clone git@github.com:MarineRoboticsGroup/PyFactorGraph.git
cd PyFactorGraph
pip install .
```

liegroups:
```bash
git clone git@github.com:utiasSTARS/liegroups.git
cd liegroups
pip install .
```

## Contributing

If you want to contribute a new feature to this package please read this brief section.

### Code Standards

Any necessary coding standards are enforced through `pre-commit`. This will run a series of `hooks` when attempting to commit code to this repo. Additionally, we run a `pre-commit` hook to auto-generate the documentation of this library to make sure it is always up to date.

To set up `pre-commit`

```bash
cd ~/manhattan-world-sim
pip3 install pre-commit
pre-commit install
```

### Testing

If you want to develop this package and test it from an external package you can also install via

```bash
cd ~/manhattan-world-sim
pip3 install -e .
```

The `-e` flag will make sure that any changes you make here are automatically translated to the external library you are working in.
