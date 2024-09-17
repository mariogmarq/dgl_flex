# Deep Gradient Leakage in FLEX

Implementation of the Deep Gradient Leakage privacy attack for federated learning (introduced in [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935)) in [FLEXible](https://github.com/FLEXible-FL/FLEXible) and Pytorch.

## Attack overview
<p align="center">
    <img src="https://github.com/mit-han-lab/dlg/blob/master/assets/method.jpg?raw=true" width="80%" />
</p>

The attack works by optimizing a random noise so it produces the same gradient as the gradient leaked. This leads to the noise being close to the original data in many cases.

## iDGL

Improved Deep Gradient Leakage is an improvement of the DGL which uses statistical knowledge about the derivative of the cross entropy in order to properly extract the label and thus improve the stability of the attack. More information in [iDLG: Improved Deep Leakage from Gradients](https://arxiv.org/abs/2001.02610).