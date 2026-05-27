
link: https://proceedings.mlr.press/v51/shahriari16.pdf

## The problem
- you have an objective function that's like a black box and is expensive to evaluate
$$ f : \mathbb{R}^m \to \mathbb{R} $$
- we aim to find the set of parameters that gives the max output of the function with the least amount of queries
$$ x \in \arg\max{f(x)} $$
-  the outputs $y$ are noise-corrupted and normally distributed around $f(x)$ with variance of $\sigma²$ 
$$ y |x \stackrel{}{\sim} \mathcal{N}(f(x), \sigma²) $$
* They don't attach the $\arg\max$ to be restricted to a subset of $\chi \subset \mathbb{R}^m$  as typical Bayesian optimization

## Bayesian Optimization
* Sequential model based approach
* maintain a surrogate model over likely functions based on observed data
* Sequentially select future points based on a policy
	* leverages the uncertainty in the surrogate model to negotiate exploration of the search space and the exploitation of currently suspected modes
	* policy is an acquisition function $\alpha_n : \mathbb{R}^m \to \mathbb{R}$
		* at iteration $n$ an input $x_{n+1}$ is selected by maximizing $\alpha_n$ then querying $f$ creating $y_{n+1}$ updating the surrogate by $(x_{n+1}, y_{n+1})$
* At the end returns the best observed set of parameters

## Gaussian Process
* Is the surrogate model for our purpose
* $GP(\mu_0, k)$
* mean function $\mu_0 : \mathbb{R}^m \to \mathbb{R}$
* kernel/convariance function $k : \mathbb{R}^m \times  \mathbb{R}^m \to \mathbb{R}$
* any finite collection of $n$ points $x_{1:n}$ the values of $f(x_1), f(x_2), \cdots, f(x_n$) are Gaussian with mean $m$ where $m_i = \mu_0(x_i)$ and $n \times n$ covariance matrix $K$ where $K_{ij} = k(x_i, x_j)$
* At any arbitrary test location $x$ we can query the model for the predicted $\hat{f}_n(x)$ conditioned on the observed data $D_n$

## Volume Doubling
* expands the search space as the optimization proceeds
* takes a initial user-defined volume, the number of iterations between expansions and a growth factor $\gamma$ . To avoid growing exponentially $\chi$ in the space $m$ the growth factor applies to the volume of $\chi$ ($\chi$ is the feasible space).

## Policy
* In general can be a specific tailored function
* In this case, the acquisition functions are based on improvement-based expected improvement approach, this one selected the next point with most expected improvement.
	* Using the surrogate model, the expectation upon a fixed $\tau$ target can be computed analytically
	* relies on an improvement function which has a target $\tau$ to improve upon
	$$ I(x) = (f(x) − \tau ) I[f(x) \ge \tau ] $$
	* when we set $\tau$ to the best output value, the function is named $\textbf{goal seeking}$. Otherwise, we could use a proxy for that value, such as the best seen value so far
		* using the best value $y^{+}$ can lead to lack of exploration, so usually a value $\zeta$ is used to improve that

![[algo.png]]


----

## What Is Surrogate Optimization?
Reference: https://www.mathworks.com/help/gads/what-is-surrogate-optimization.html
