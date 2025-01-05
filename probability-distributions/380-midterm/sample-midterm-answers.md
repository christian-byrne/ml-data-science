## Question 1

### Part (a)

> Sample mean is an unbiased estimator for probability of heads $p$ of a Bernoulli distribution.

#### Answer

`True`

#### Explanation

An **estimator** is considered unbiased if its expected value is equal to the parameter it estimates. For a Bernoulli random variable $X$ with probability $p$ of success (heads), the probability of success is also the mean $E(X) = p$. Given a sample $X_1, X_2, \ldots, X_n$ from a Bernoulli distribution, the sample mean is:

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

Since $E(X_i) = p$ for each $X_i$, by the **linearity of expectation**:

$$
E(\bar{X}) = \frac{1}{n} \sum_{i=1}^{n} E(X_i) = \frac{1}{n} \cdot n \cdot p = p
$$

Thus, the sample mean $\bar{X}$ is an unbiased estimator of $p$.

---

### Part (b)

> $\frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2$ is an unbiased estimator of variance.

#### Answer

`False`

#### Explanation

The expression $\frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2$ is known as the **sample variance** without Bessel’s correction. For an unbiased estimator of the population variance $\sigma^2$, we use:

$$
\text{Unbiased sample variance} = \frac{1}{n - 1} \sum_{i=1}^{n} (X_i - \bar{X})^2
$$

This correction factor of $n - 1$ (known as **Bessel's correction**) is necessary because $\bar{X}$ is itself a random variable and has absorbed some of the variability of $X$. The **expected value** of $\frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2$ is actually $\frac{n - 1}{n} \sigma^2$, not $\sigma^2$, hence it is a biased estimator of the population variance.



---


### Part (c)

> Let $X_1, X_2, \ldots, X_n \sim \text{Bernoulli}(\theta)$. The maximum likelihood estimator for unknown $\theta$ is the same as the sample mean.

#### Answer

`True`

#### Explanation

For $X_1, X_2, \ldots, X_n \sim \text{Bernoulli}(\theta)$, the probability mass function is:

$$
P(X = x) = \theta^x (1 - \theta)^{1 - x}
$$

The **likelihood function** for $\theta$ given $X_1, X_2, \ldots, X_n$ is:

$$
L(\theta) = \prod_{i=1}^{n} \theta^{X_i} (1 - \theta)^{1 - X_i} = \theta^{\sum X_i} (1 - \theta)^{n - \sum X_i}
$$

Taking the log-likelihood:

$$
\ln L(\theta) = \left( \sum X_i \right) \ln \theta + \left( n - \sum X_i \right) \ln (1 - \theta)
$$

Differentiating with respect to $\theta$ and setting to zero, we find:

$$
\hat{\theta} = \frac{\sum X_i}{n} = \bar{X}
$$

Thus, the maximum likelihood estimator (MLE) for $\theta$ is the sample mean $\bar{X}$.

---


### Part (d)

> Weak law of large numbers claims that as the size of sample increases, the sample mean converges to the distribution mean in probability.

#### Answer

**True**

#### Explanation

The **Weak Law of Large Numbers** (WLLN) states that for a sequence of independent and identically distributed (i.i.d.) random variables $X_1, X_2, \ldots, X_n$ with mean $\mu$ and finite variance $\sigma^2$, the sample mean $\bar{X}_n = \frac{1}{n} \sum_{i=1}^{n} X_i$ converges to the population mean $\mu$ **in probability** as $n \to \infty$:

$$
\forall \epsilon > 0, \lim_{n \to \infty} P(|\bar{X}_n - \mu| < \epsilon) = 1
$$

This means that as $n$ increases, the probability that the sample mean $\bar{X}_n$ deviates from $\mu$ by any positive amount $\epsilon$ approaches zero, confirming convergence in probability.


---


### Part (e)

> The quantile function is the inverse of the pdf.

#### Answer

**False**

#### Explanation

The **quantile function** $Q(p)$ for a distribution with cumulative distribution function (CDF) $F(x)$ is the inverse of the CDF, not the probability density function (PDF). It is defined as:

$$
Q(p) = F^{-1}(p)
$$

where $p \in [0,1]$ and $F(x)$ is the probability that a random variable $X$ takes on a value less than or equal to $x$.

The PDF, on the other hand, represents the rate of change of the CDF:

$$
f(x) = \frac{d}{dx} F(x)
$$

Therefore, the quantile function does not directly relate to the PDF but rather to the CDF.


---


### Part (f)

> Sampling $n$ balls with replacement from a box containing $R$ red balls and $B$ blue balls can be modeled with a hypergeometric distribution.

#### Answer

**False**

#### Explanation

The **hypergeometric distribution** models the probability of a certain number of successes in a sequence of draws **without replacement** from a finite population. The probability mass function of a hypergeometric random variable $X$, representing the number of red balls drawn from a population of $R + B$ balls (with $R$ red and $B$ blue), is:

$$
P(X = k) = \frac{\binom{R}{k} \binom{B}{n - k}}{\binom{R + B}{n}}
$$

However, **sampling with replacement** results in a **binomial distribution**. For $n$ trials with replacement, where each trial has a probability $p = \frac{R}{R + B}$ of drawing a red ball, the number of red balls drawn follows a binomial distribution:

$$
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
$$

Thus, the hypergeometric distribution does not apply to sampling with replacement.


## Question 2

> The exponential distribution has cdf $1 - e^{−λx}$. Describe how to sample two random points from this distribution using inverse transform sampling

---

**Goal of Inverse Transform Sampling**: 

Let $X$ be a random variable whose distribution can be described by a CDF $F(x)$. We want to generate values of $X$, but for them to be valid they should be distributed according to $F(x)$, otherwise they won't be realistically distributed the way the real values of $X$ are. 

**Method of Inverse Transform Sampling**: 

1. Find the inverse of the CDF $F(x)$, which is $F^{-1}(x)$.
2. Generate a random value $U$ from a uniform distribution on $[0, 1]$.
    - Since the range of probabilities is $[0, 1]$, generating a random number from $U$ can be accurately done.
3. Plug $U$ into the inverse CDF $F^{-1}(U)$.
    - This will give you a value of $X$ associated with the probability $U$.
    - When done multiple times, the values of $X$ generated will be distributed according to $F(x)$.
  
**Applied to $cdf(x) = 1 - e^{-\lambda x}$**:

1. Find the inverse of $F(x)$:
   1. $y = 1 - e^{-\lambda x} \implies 1 - y = e^{-\lambda x} \implies x = -\frac{1}{\lambda} \ln(1 - y)$
2. Generate two random values $U_1, U_2$ from a uniform distribution on $[0, 1]$. Say $U_1 = 0.5$ and $U_2 = 0.7$.
3. Plug $U_1, U_2$ into the inverse CDF:
   1. $F^{-1}(0.5) = -\frac{1}{\lambda} \ln(1 - 0.5) = -\frac{1}{\lambda} \ln(0.5) = \frac{\ln(2)}{\lambda}$
   2. $F^{-1}(0.7) = -\frac{1}{\lambda} \ln(1 - 0.7) = -\frac{1}{\lambda} \ln(0.3) = \frac{\ln(3)}{\lambda}$
4. The values of $X$ associated with the probabilities $0.5, 0.7$ are $\frac{\ln(2)}{\lambda}, \frac{\ln(3)}{\lambda}$. The distribution of $U_1, U_2$ will be according to the exponential distribution.

## Question 3

### Part (a)

> Associate each plot below with one of the Pearson correlation coefficient values: $-0.4, 0.8, -0.8, -1, 1, 0, 0.4$


---

...


### Part (b)

> Discuss briefly what the phrase correlation does not imply causation means and provide an example

----


"Correlation does not imply causation" means that just because two variables are statistically associated, it doesn’t mean one causes the other. For example, ice cream sales and drowning incidents are correlated because both increase during hot weather, but one does not cause the other.



### Part (c)

> Discuss what Z-score normalization is and why it is useful in handling data compatibility issues while doing data cleaning.

----

Z-score normalization is a technique used to adjust data to have a mean of zero and a standard deviation of one. It is calculated as $Z = \frac{X - \mu}{\sigma}$, where $X$ is the data point, $\mu$ is the mean, and $\sigma$ is the standard deviation.

Z-score normalization is useful in handling data compatibility issues during data cleaning because it brings all features to a similar scale. This is essential for algorithms that are sensitive to feature scales. By normalizing the data, we ensure that all features contribute equally to the analysis, preventing any one feature from dominating the results due to differences in scale.




### Part (d)


> Describe why we need multiple testing correction and a method to resolve it


---


**Explanation**

When conducting multiple tests, the likelihood of false positives increases. Because although for a given significance level $\alpha$ the probability of a Type I error is $\alpha$, the probability of at least one Type I error increases with the number of tests.

Imagine we have 100 independent tests, each with a significance level of 0.05. The probability of at least one Type I error is $1 - (1 - 0.05)^{100} \approx 0.994$. This is much higher than the 0.05 significance level we intended.

**Example** 

You are conducting a roadside breathalyzer test for alcohol. The test is 95% accurate, meaning that 5% of the time, it will give a false positive. I.e., the $\alpha$ level is 0.05 because the probability of a false positive (rejecting the $H_0$ of *not drunk* when $H_0$ is actually true) is 0.05. 

Imagine that the *true* rate of drunk driving (proportion of people on road at any given time that are drunk) is .002. If you test 1000 people, you'd expect 2 of them to be drunk.

However, since your probability of a Type I error is 0.05, you'd expect 5% of the 1000 non-drunk people (50 people) to test positive for drunk driving.

Thus, among the people who go to jail for drunk driving, only 2 out of the 52 are actually drunk. The other 50 are false positives.

**Solution**

Multiple testing correction, like the Bonferroni correction (dividing significance by the number of tests), helps reduce this risk, maintaining an accurate family-wise error rate. 

It does this by adjusting the significance level for each test to ensure that the overall probability of a Type I error across all tests remains at the desired level. This correction helps control the false positive rate and ensures that the results are reliable and not due to chance.




## Question 4


### Part (a)


> Suppose that a baseball hitter has a probability of success $p = 0.7$. What is the probability that she hits (has success) more than 6 times out of a total of $15$ throws. You may make use of the following outputs from `scipy.stats`: `binom.pmf(5,15,0.7) = 0.003`, `binom.cdf(5,15,0.7) = 0.004`, `binom.pmf(6,15,0.7) = 0.012`, `binom.cdf(6,15,0.7) = 0.015`.

---


For a hitter with a probability of success $p = 0.7$ over 15 throws, we want $P(X > 6)$.

Using the binomial CDF:

$$
P(X > 6) = 1 - P(X \leq 6) = 1 - 0.015 = 0.985
$$

Since CDF finds the probability mass *up to* a certain point, we subtract the CDF of 6 from 1 to find the probability mass *after* 6.

**Answer:** $P(X > 6) = 0.985$


### Part (b)

> Suppose that $X$ has the normal distribution with mean $5$ and standard deviation $2$. Find out the value of $P(1 < X < 8)$. You may make use of the following outputs from `scipy.stats`: `norm.pdf(2) = 0.054`, `norm.cdf(2) = 0.977`, `norm.pdf(3) = 0.004`, `norm.cdf(1.5) = 0.933`.

----

Given a normal distribution with mean $\mu = 5$ and standard deviation $\sigma = 2$, we want to find $P(1 < X < 8)$.

The CDF/PDF of the normal distribution expects a *normal* distribution. So, we first have to *normalize* the values 1 and 8 to the standard normal distribution. This can be done by finding the values' z-scores:

$$
z = \frac{x - \mu}{\sigma}
$$

For $x = 1$:

$$
z_1 = \frac{1 - 5}{2} = -2
$$

For $x = 8$:

$$
z_8 = \frac{8 - 5}{2} = 1.5
$$

If we consider the inequality range on a number line, we have:  

```
——— 1 —————————— 5 ————————— 8 —————
_____________________________| cdf(8)
____| cdf(1)
```

We can see that cdf between 1 and 8 is the cdf at 8 minus the cdf at 1:

$$
P(1 < X < 8) = P(X < 8) - P(X < 1) = \text{cdf}(1.5) - \text{cdf}(-2) = 0.933 - 0.023 = 0.91
$$



## Question 5

### Part (a)

> Two boxes contain long bolts and short bolts. Suppose that one box contains $60$ long bolts and $40$ short bolts, and that the other box contains $10$ long bolts and $20$ short bolts. Suppose also that one box is selected at random and a bolt is then selected at random from that box. What is the probability that this bolt is long?



- $P(B_1) = \frac{1}{2}$
- $P(B_2) = \frac{1}{2}$
- $P(\text{Long} | B_1) = 0.6$
- $P(\text{Long} | B_2) = \frac{1}{3}$

Using Law of Total probability:

$$
P(\text{Long}) = P(\text{Long} | B_1) \cdot P(B_1) + P(\text{Long} | B_2) \cdot P(B_2) = 0.3 + 0.1665 = 0.4665
$$

**Answer:** $P(\text{Long}) \approx 0.467$



### Part (b)

> Suppose that $X$ and $Y$ are random variables such that $Var(X) = 9$, $Var(Y) = 4$, and the Pearson correlation coefficient of the two variables $r = −1/6$. Determine $Var(X + Y)$

---


The formula for the variance of the sum is:


$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2 \, \text{Cov}(X, Y)
$$

We calculate $\text{Cov}(X, Y)$ using the correlation coefficient:

$$
\text{correalation coefficient} = r = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X) \cdot \text{Var}(Y)}} \implies
$$

$$
\text{Cov}(X, Y) = r \cdot \sqrt{\text{Var}(X) \cdot \text{Var}(Y)} =
$$

$$
-\frac{1}{6} \cdot \sqrt{9 \cdot 4} = \plusmn 1
$$

However, since the correlation coefficient is negative, it must be the case that the covariance is negative. So we can remove the $\plusmn$ sign and use $-1$.

Substitute back:

$$
\text{Var}(X + Y) = 9 + 4 + 2 \cdot (-1) = 11
$$

**Answer:** $\text{Var}(X + Y) = 11$


## Question 6

### Part (a)

> We take a random sample of size $64$ from a distribution with unknown mean $μ$ that has standard deviation $σ = 4$. Assume the sample mean is $6$. What is the $90\%$ confidence interval for mean $μ$? You may make use of the following outputs from `scipy.stats`: `norm.ppf(0.995) = 2.58`, `norm.ppf(0.95) = 1.64`, `norm.cdf(0.995) = 0.84`, `norm.cdf(0.95) = 0.83`

---


The population mean is unknown, but since we know the population standard deviation, we can use the z-distribution for the critical value.

Critical z-score for 90% confidence: z-score such that $P(Z < z) = 0.95$ = `norm.ppf(0.95) = 1.64` 

Calculate Standard error: 

$$
\text{SE} = \frac{\sigma}{\sqrt{n}} = \frac{4}{\sqrt{64}} = 0.5
$$


Calculate Margin of error (ME):

$$
\text{ME} = \text{critical value} \cdot \text{SE} = 1.64 \cdot 0.5 = 0.82
$$

The confidence interval is found from the formula:

$$
(\bar{x} - \text{ME}, \bar{x} + \text{ME})
$$

Substitute:

$$
(6 - 0.8225, 6 + 0.8225) = (5.18, 6.82)
$$

**Answer:** The 90% confidence interval for $\mu$ is $(5.18, 6.82)$.


### Part (b)

> We take a random sample of size $4$ from a distribution with unknown mean $μ$ and unknown $σ$. The values of the sample are $3, 5, 2, 6$. What is the $90\%$ confidence interval for mean $μ$? You may make use of the following outputs from `scipy.stats`: `norm.ppf(0.95) = 1.64`, `t.ppf(0.95,3) = 2.35`, `norm.ppf(0.995) = 2.58`, `t.ppf(0.995,3) = 5.84`

---

Since the population standard deviation is unknown, we use the t-distribution for the critical value.


- Sample mean $\bar{x} = 4$
- Sample standard deviation $s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2} = 1.87$
- Degrees of freedom $n - 1 = 3$
- Critical value is value of $t$ such that $P(T < t) = 0.95$ = `t.ppf(0.95,3) = 2.35`

Calculate Standard error:

$$
\text{SE} = \frac{s}{\sqrt{n}} = \frac{1.87}{\sqrt{4}} = 0.935
$$

Calculate Margin of error (ME):

$$
\text{ME} = \text{critical value} \cdot \text{SE} = 2.35 \cdot 0.935 = 2.19
$$


Find confidence interval:

$$
(4 - 2.15, 4 + 2.15) = (1.85, 6.15)
$$

**Answer:** The 90% confidence interval for $\mu$ is $(1.85, 6.15)$.
