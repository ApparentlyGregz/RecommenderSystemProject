# Toy simulations (synthetic data)

These scripts motivated two design choices. They use made-up data, not EdNet.

- `beta_sim.py`: with random negatives and time gaps measured in the same window, the forgetting weight beta is learned with the wrong sign (about -1.2). With labels from the next window and attempted-but-correct negatives, it comes out positive (about +0.6).
- `eval_sim.py`: ranking all topics rewards the topics the app happened to show, not the risk of failing. A model that knows the true risk scores lower than most-popular in that setting; ranking only attempted topics fixes it.
- `grad_check.py`: the learned half-life receives gradients through the normalised sparse propagation.

Run with `python sim/<script>.py` (PyTorch needed for `beta_sim.py` and `grad_check.py`).
