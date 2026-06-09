URL: https://arxiv.org/pdf/1711.10781

* Given a matrix $M$ we would like to approximate it to $\hat{M}$ with a lower rank
	* this is not unique
		* can be unique under certain constraints
	* is the same as minimizing the norm between $\min || M - \hat{M} ||$ being $\hat{M} = AB^{T}$
* rotation problem
	* Matrix $R$ and $R^{-1}$ are added between $AB^{T}$
		* $\hat{M} = ARR^{⁻1}B^T = (AR)(BR^{-T})^T = \tilde{A}\tilde{B}^T$
* tensor uniqueness
	* exists only one combination of rank-1 tensors outer-product that sum up to a high order tensor under a common scaling factor