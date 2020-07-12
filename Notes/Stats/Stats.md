# Stats Notes





### Bernoulli random variable

Any random variable whose only possible values are 0 and 1 is called a
Bernoulli random variable. 

assuming independence and constant probabilities 



-----------

[source](<https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119197096.app03>)

## DISCRETE DISTRIBUTIONS

> best books is stat 285 probability_and_statistics_for_engineering_and_the_sciences by Devore



### conditional distribution 

Imagine a bar graph of categories, each with a probability.... ya that’s it! 

![enter image description here](https://i.stack.imgur.com/gJxXi.jpg)

### Marginal distribution 

Using the above example, the n-tuple gets combined. The two stratum/ ‘types of speakers’ are considers a single entity 

so, A would be 88.75%. 



### Discrete Uniform

The discrete uniform distribution is also known as the equally likely outcomes distribution, where the distribution has a set of N elements, and each element has the same probability. This distribution is related to the uniform distribution, but its elements are discrete and not continuous.



###  The Poisson Distribution

The Poisson distribution describes the number of times an event occurs in a given
interval, such as the number of telephone calls per minute or the number of errors
per page in a document.
The three conditions underlying the Poisson distribution are:

1. The number of possible occurrences in any interval is unlimited.
2. The occurrences are independent. The number of occurrences in one interval
    does not affect the number of occurrences in other intervals.
3. The average number of occurrences must remain the same from interval to
    interval.



$$
f(x,\lambda) = \lambda e^{-\lambda} \div x!
\\
----------
\\
\lambda\ =  occurrences\ expected\  over\ our\ time\ peroid = mean  
\\
x = \ number\ of\ events
\\
STD = \sqrt{\lambda}
\\
skewness = \frac {1}{\sqrt{\lambda}}
//
$$



Rate > 0 and <= 1,000 (that is, 0.0001 <= rate <= 1,000)

>  so, if we were told to expect 5 meteors per hour on average.
>
> p(3 meteors in 1 hour) = ( 5^3) * e ^ -5  / 3!   = 0.140373896



### Binomial Distribution 



The binomial distribution describes the number of times a particular event occurs
in a fixed number of trials, such as the number of heads in 10 flips of a coin or the
number of defective items out of 50 items chosen.
The three conditions underlying the binomial distribution are:

1. For each trial, only two outcomes are possible that are mutually exclusive.
2. The trials are independent—what happens in the first trial does not affect the
     next trial.
3. Each trial can result in one of the same two possible outcomes (dichotomous
     trials), which we generically denote by success (S) and failure (F). 
4. The probability of an event occurring remains the same from trial to trial.



$$
P(x) = \dfrac {n!} {x!(n- x)!} \ p^ x(1-p)^{(n-x)}
\\ for\  n > 0; x=0, 1, 2, n; \ and \ 0<p<1
$$




### Geometric Distribution

The geometric distribution describes the number of trials until the first successful
occurrence, such as the number of times you need to spin a roulette wheel before
you win.
The three conditions underlying the geometric distribution are:
1. The number of trials is not fixed.
2. The trials continue until the first success.
3. The probability of success is the same from trial to trial.
    The mathematical constructs for the geometric distribution are as follows:

> The mathematical constructs for the geometric distribution are as follows:

$$
 p(x) = p(1-p)^{x-1}
 \\
 for\ 0 < p < 1 \ and \ x = 1,2,3...n
$$




$$
Mean = \frac 1 p - 1  

\\

std = \sqrt{ \frac{1-p}{P^2}}
$$




## CONTINUOUS DISTRIBUTIONS



### Chi-Square Distribution

The chi-square distribution is a probability distribution used predominantly in hypothesis testing, and is related to the gamma distribution and the standard normal distribution. For instance, the sums of independent normal distributions are distributed as a chi-square( X^2) with k degrees of freedom 

### Exponential Distribution

The exponential distribution is widely used to describe events recurring at random
points in time, such as the time between failures of electronic equipment or the time between arrivals at a service booth. It is related to the Poisson distribution, which describes the number of occurrences of an event in a given interval of time. An important characteristic of the exponential distribution is the “memoryless” property, which means that the future lifetime of a given object has the same distribution, regardless of the time it existed. In other words, time has no effect on future outcomes. The mathematical constructs for the exponential distribution are as follows:
$$
f(x) = \lambda e^{-\lambda x }
\\
-
\\
mean = \frac {1}{\lambda}
\\
STD = \frac {1}{\lambda}
$$

> for x and lambda  >= 0

### Gamma Distribution (Erlang Distribution)

The gamma distribution applies to a wide range of physical quantities and is related to other distributions: lognormal, exponential, Pascal, Erlang, Poisson, and chisquare. It is used in meteorological processes to represent pollutant concentrations and precipitation quantities. The gamma distribution is also used to measure the time between the occurrence of events when the event process is not completely random. Other applications of the gamma distribution include inventory control, economic theory, and insurance risk theory. The gamma distribution is most often used as the distribution of the amount of time until the rth occurrence of an event in a Poisson process. When used in this fashion, the three conditions underlying the gamma distribution are:

1. The number of possible occurrences in any unit of measurement is not limited to a fixed number.
2. The occurrences are independent. The number of occurrences in one unit of
    measurement does not affect the number of occurrences in other units.
3. The average number of occurrences must remain the same from unit to unit.
    The mathematical constructs for the gamma distribution are as follows:

### Logistic Distribution

The logistic distribution is commonly used to describe growth, that is, the size of a
population expressed as a function of a time variable. It also can be used to describe chemical reactions and the course of growth for a population or individual.

 ### Lognormal Distribution

The lognormal distribution is widely used in situations where values are positively
skewed, for example, in financial analysis for security valuation or in real estate for
property valuation, and where values cannot fall below zero.
Stock prices are usually positively skewed rather than normally (symmetrically)
distributed. Stock prices exhibit this trend because they cannot fall below the lower
limit of zero but might increase to any price without limit. Similarly, real estate
prices illustrate positive skewness and are lognormally distributed as property values cannot become negative.

The three conditions underlying the lognormal distribution are:

1. The uncertain variable can increase without limits but cannot fall below zero.
2. The uncertain variable is positively skewed, with most of the values near the
    lower limit.
3. The natural logarithm of the uncertain variable yields a normal distribution.





### Normal Distribution

The normal distribution is the most important distribution in probability theory because it describes many natural phenomena, such as people’s IQs or heights. Decision makers can use the normal distribution to describe uncertain variables such as the inflation rate or the future price of gasoline.
The three conditions underlying the normal distribution are:

1. Some value of the uncertain variable is the most likely (the mean of the
   distribution).
2. The uncertain variable could as likely be above the mean as it could be below
   the mean (symmetrical about the mean).
3. The uncertain variable is more likely to be in the vicinity of the mean than further away.



### Pareto Distribution

The Pareto distribution is widely used for the investigation of distributions associated with such empirical phenomena as city population sizes, the occurrence of natural resources, the size of companies, personal incomes, stock price fluctuations, and error clustering in communication circuits.





### Student’s t Distribution

The Student’s t distribution is the most widely used distribution in hypothesis testing. This distribution is used to estimate the mean of a normally distributed population when the sample size is small, and is used to test the statistical significance of the difference between two sample means or confidence intervals for small sample sizes.





### Triangular Distribution

The triangular distribution describes a situation where you know the minimum,
maximum, and most likely values to occur. For example, you could describe the
number of cars sold per week when past sales show the minimum, maximum, and
usual number of cars sold. The three conditions underlying the triangular distribution are:

1. The minimum number of items is fixed.
2. The maximum number of items is fixed.
3. The most likely number of items falls between the minimum and maximum
    values, forming a triangular-shaped distribution, which shows that values near
    the minimum and maximum are less likely to occur than those near the most
    likely value.



### Triangular Distribution

The triangular distribution describes a situation where you know the minimum,
maximum, and most likely values to occur. For example, you could describe the
number of cars sold per week when past sales show the minimum, maximum, and
usual number of cars sold. The three conditions underlying the triangular distribution are:

1. The minimum number of items is fixed.
2. The maximum number of items is fixed.
3. The most likely number of items falls between the minimum and maximum
    values, forming a triangular-shaped distribution, which shows that values near
    the minimum and maximum are less likely to occur than those near the most
    likely value.





### Uniform Distribution
With the uniform distribution, all values fall between the minimum and maximum
and occur with equal likelihood. The three conditions underlying the uniform distribution are:

1. The minimum value is fixed.
2. The maximum value is fixed.
3. All values between the minimum and maximum occur with equal likelihood.





### Weibull Distribution (Rayleigh Distribution)

The Weibull distribution describes data resulting from life and fatigue tests. It is
commonly used to describe failure time in reliability studies as well as the breaking
strengths of materials in reliability and quality control tests. Weibull distributions
are also used to represent various physical quantities, such as wind speed.
The Weibull distribution is a family of distributions that can assume the properties of several other distributions. For example, depending on the shape parameter you define, the Weibull distribution can be used to model the exponential and
Rayleigh distributions, among others. The Weibull distribution is very flexible. When the Weibull shape parameter is equal to 1.0, the Weibull distribution is identical to the exponential distribution. The Weibull location parameter lets you set up an exponential distribution to start at a location other than 0.0. When the shape parameter is less than 1.0, the Weibull distribution becomes a steeply declining curve. A manufacturer might find this effect useful in describing part failures during a burn-in period.







----------

## Distribution Functions






### Cumulative distribution function



$$
{\displaystyle F_{X}(x)=\operatorname {P} (X\leq x)}
\\
 {\displaystyle \operatorname {P} (a<X\leq b)=F_{X}(b)-F_{X}(a)}
 
 
$$
![Image result for exponential distribution cdf pdf](https://www.causeweb.org/cause/archive/repository/statjava/images/exp_table.png)

Shows the increasing quantity over time 

CDF of a normal distribution![Image result for cdf normal](https://www.itl.nist.gov/div898/handbook/eda/section3/gif/norcdf.gif)









### Probability density function

The probability density function (PDF) P(x) of a continuous distribution is defined as the derivative of the (cumulative) [distribution function](http://mathworld.wolfram.com/DistributionFunction.html) 

PDF of a normal distribution is a normal distribution with the same mean and STD 







































------------------

## Inequities



### Chebyshev's Inequality



![[eq2]](https://www.statlect.com/images/Chebyshev-inequality__6.png)

P (| random variable - mean| >= K ) <= variance / K^2

> Suppose we extract an individual at random from a population whose members have an average income of $40,000, with a standard deviation of $20,000. What is the probability of extracting an individual whose income is either less than $10,000 or greater than $70,000? In the absence of more information about the distribution of income, we cannot compute this probability exactly. However, we can use Chebyshev's inequality to compute an upper bound to it. If ![X](https://www.statlect.com/s.gif) denotes income, then ![X](https://www.statlect.com/s.gif) is less than $10,000 or greater than $70,000 if and only if

P (| random variable - mean| >= K ) <= variance / K^2

K = 30k; since 10k + K = mean; mean + K = 70k 

​							<= 20k ^2 / 30k^2 

Therefore, the probability of extracting an individual outside the income range $10,000-$70,000 is less than 4/9 





### Hoeffding's Inequality




### Hoeffding's lemma


$$
E[ e^{\lambda X}] \leq  exp( \lambda^2 (b-a)^2 \div 8 )
$$
