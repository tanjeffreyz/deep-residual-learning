<h1 align="center">Deep Residual Learning for Image Recognition</h1>
PyTorch implementations of the deep residual networks published in "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.



<h2>Notes</h2>
**Anatomy of a residual block**

            X -----------
            |           |
        weight layer    |
            |           |
        weight layer    |
            |           |
           (+) <---------
            |
           H(X)

This entire block describes the underlying mapping H(X) = F(X) + X where F is the mapping
described by the two weight layers. Rearranging yields F(X) = H(X) - X. This shows that,
instead of directly mapping an input X to an output H(X), the weight layers are responsible
for describing what to change, if anything, about the input X to reach the desired mapping
H(X).

Intuitively, it is easier to modify an existing function than to create a brand new one
from scratch.



<h2>References</h2>

[[1](https://arxiv.org/abs/1512.03385)] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*. arXiv:1512.03385v1 [cs.CV] 10 Dec 2015.
