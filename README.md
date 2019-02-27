# MechaModLearn

Authors: [@kunalmenda](https://github.com/kunalmenda) and [@rejuvyesh](https://github.com/rejuvyesh)

This library provides a structured framework for learning mechanical systems in PyTorch.

---

## Installation

Requires Python3.

```
git clone https://github.com/sisl/mechamodlearn.git
cd mechamodlearn
pip install -e .
```

## Usage
Example experiments are placed in [`experiments`](./experiments) directory.

To run the Simple experiment:

```
python experiments/simple.py --logdir /tmp/simplelog
```

## References

---
If you found this library useful in your research, please consider citing our [paper](https://arxiv.org/abs/1902.08705):
```
@article{gupta2019mod,
    title={A General Framework for Structured Learning of Mechanical Systems},
    author={Gupta, Jayesh K. and Menda, Kunal and Manchester, Zachary and Kochenderfer, Mykel},
    journal={arXiv preprint arXiv:1902.08705},
    year={2019}
}
```
