# Think DSP 

## Digital Signal Processing in Python



> Signal processing is one of my favorite topics. It is useful in many areas of
> science and engineering, and if you understand the fundamental ideas, it
> provides insight into many things we see in the world, and especially the
> things we hear. 
>
> With a programming-based approach, I can present the most important
> ideas right away. By the end of the first chapter, you can analyze sound
> recordings and other signals, and generate new sounds. Each chapter introduces a new technique and an application you can apply to real signals. At
> each step you learn how to use a technique first, and then how it works.
> This approach is more practical and, I hope you’ll agree, more fun. 

## 1. Sounds and signals 

The Discrete Fourier Transform (DFT) takes a signal and produces its spectrum.

The Fast Fourier Transform, or FFT, which is an efficient way to compute the DFT 

A Wave represents a signal evaluated at a sequence of points in time 

The primary classes in thinkdsp are Signal, Wave, and Spectrum. Given a
Signal, you can make a Wave. Given a Wave, you can make a Spectrum 

A Wave object contains three attributes:

* ys is a NumPy array that contains the values in the signal

* ts is an array of the times where the signal was
  evaluated or sampled

* framerate is the number of samples per unit of
  time 

  ```python
  wave.scale(2)
  wave.shift(1)
  ```

  The evaluate function, y = A cos(2pi f t + **phi**_0  ) 

## 2. Harmonics

types of waves: 

* Triangle
* Square 
* Aliasing 

When you evaluate the signal at discrete points in time,
you lose information about what happened between samples.  

```np.fft``` is the NumPy module that provides functions related to the Fast
Fourier Transform (FFT) 

- worthy of mention, book states, Later we’ll see the full FFT, which can
  handle complex signals (see Section 7.9)*

## 3. Non-periodic signals 

Thinkdsp provides a Signal called Chirp that makes a sinusoid that sweeps linearly
through a range of frequencies.

 ``` python
wave = signal.make_wave() 
signal = thinkdsp.Chirp(start=220, end=880, amp=1.0)
 ```

## 4. Noise 



When the phase, increases linearly over time 

​		*phi* = 2pi f t

When frequency is a function of time, the change in phase during a short
time interval, Dt, is: 

​		*delta-phi* = 2pi f(t) Delta-t

![1555996674677](.\images\freq and phase.JPG)

* Frequency is the derivative of phase 

* Phase is the integral of frequency  

  

​	The interval we hear between two notes depends on the
ratio of their frequencies, not the difference. “Interval” is the musical term
for the perceived difference between two pitches. 

​	For example, an octave is an interval where the ratio of two pitches is 2. So
the interval from **220 to 440 is one octave and the interval from 440 to 880**
is also one octave. The difference in frequency is bigger, but the ratio is the
same. 

​	As a result, if frequency increases linearly, as in a linear chirp, the perceived
pitch increases logarithmically.

​	If you want the perceived pitch to increase linearly, the frequency has to
increase exponentially. A signal with that shape is called an exponential
chirp 

```python
signal = thinkdsp.Chirp(start=220, end=440)
wave = signal.make_wave(duration=1)
spectrum = wave.make_spectrum()
```

​	To recover the relationship between frequency and time, we can break the
chirp into segments and plot the spectrum of each segment. The result is called a short-time Fourier Transform (STFT) 

FFT is most efficient when the number of samples is a power of 2.

* Time resolution of the spectrogram is the duration of the segments,
  which corresponds to the width of the cells in the spectrogram 
* Frequency resolution is the frequency range between elements in the
  spectrum, which corresponds to the height of the cells 

The time resolution, n/r, is the inverse of frequency resolution, r/n. So if
one gets smaller, the other gets bigger. 

​	One common problem is **discontinuity** at the beginning and end of the segment. Because DFT assumes that the signal is periodic, it implicitly connects the end of the segment back to the beginning to make a loop. If the end does not connect smoothly to the beginning, __the discontinuity creates additional frequency components__ in the segment that are not in the signal. 

A “window” is a function designed to transform a non-periodic segment
into something that can pass for periodic. 



## 4. Noise

> “Noise” also refers to a signal that contains components at many frequencies, so it lacks the harmonic structure of the periodic signals we
> saw in previous chapters. 



### There are at least three things we might like to know about a noise signal or its spectrum: 

* Distribution: The distribution of a random signal is the set of possible
  values and their probabilities.
  *  For example, in the uniform noise signal, the set of values is the range from -1 to 1, and all values have the same probability. 
  * An alternative is Gaussian noise, where the set of values is the range from negative to positive infinity, but values near 0 are the most likely, with probability that drops off according to the Gaussian or “bell” curve. 
* Correlation: Is each value in the signal independent of the others, or
  are there dependencies between them? 
  * In UU noise, the values are independent. 
  * An alternative is Brownian noise, where each value is
    the sum of the previous value and a random “step”. So if the value
    of the signal is high at a particular point in time, we expect it to stay
    high, and if it is low, we expect it to stay low. 
* Relationship between power and frequency: In the spectrum of UU
  noise, the power at all frequencies is drawn from the same distribution; that is, the average power is the same for all frequencies. 
  * An alternative is pink noise, where power is inversely related to frequency; 

**Power is the square of Amplitude**



UU noise is uncorrelated 

* An alternative is Brownian noise, in which each value is the
  sum of the previous value and a random “step”. 
  * Brownian motion is often described using a random
    walk, which is a mathematical model of a path where the distance between
    steps is characterized by a random distribution. 

For Brownian noise, the slope of the power spectrum is -2  

​		```			log P = k - 2 log f ```

where P is power, f is frequency, and k is the intercept of the line, which is
not important for our purposes. Exponentiating both sides yields: 

​		```P = K/ f^2 ``` ;K = e^k, but is not important 

More relevant is that power is proportional to 1/ f^2 

There is nothing special about the exponent 2. More generally, we can synthesize noise with any exponent, b. 

​		``` P = K/ f^b ``` 

* When b = 0, power is constant at all frequencies, so the result is white noise.

* When b = 2 the result is red noise.

* When b is between 0 and 2, the result is between white and red noise, so it
  is called pink noise. 

  

**Beta is the desired exponent** 

```javascript
signal = thinkdsp.BrownianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
wave.plot() 
```



### Gaussian noise 

np.random.normal returns a NumPy array of values from a Gaussian distribution, in this case with mean 0 and standard deviation self.amp. In theory
the range of values is from negative to positive infinity, but we expect about
99% of the values to be between -3 and 3 

To illustrate.

```
thinkstats2.NormalProbabilityPlot(spectrum.imag) 
thinkstats2.NormalProbabilityPlot(spectrum.real)
```

is a graphical way to test whether a distribution is Gaussian. 

## 5 Autocorrelation 

## 6. Discrete Cosine Transform 

## 7. Discrete Fourier Transform 

## 8. Filtering and Convolution

## 9. Differentiation and Integration 

## 10. LTI systems 

## 11. Modulation and sampling 











