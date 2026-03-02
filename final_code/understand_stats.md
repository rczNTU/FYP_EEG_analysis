

EEG data is just Voltage over time
Time (ms) →   0   2   4   6   8   ...
Voltage (µV) → 3  -5   2   8  -1  ...

(1)
we cut these data into trials.
then we ask:How strong is 40 Hz inside this signal?
To do this:
Compute PSD (power spectral density)
Or FFT
Voltage over time -> Power at each frequency
Frequency → Power
38 Hz → 1.01
39 Hz → 1.12
40 Hz → 1.25
41 Hz → 1.08

(2)
Reduce Each Trial to ONE Number.
For each trial, you extract one summary value.
Eg.
find Mean power between 35–45 Hz. 
trial 1 baseline->Gamma power = 1.03, 
trial 1 pattern: Gamma power = 1.21

DO this for all trials.
Bseline Pattern
1.03    1.21
0.98.   1.18
...      ...

then we ask Is the average of these differences significantly above 0?Null hyphothtes:The true average difference = 0
AKA.Is the difference big enough that random noise is unlikely to explain it?(Is it reliably larger across trials?)

==================
Baseline:
1.05, 0.98, 1.02, 1.10...

Pattern:
1.20, 1.15, 1.30, 1.10...
Is Pattern REALLY higher?
Or is this just random fluctuation?

t-test:
Are the averages different enough compared to how noisy the data is?
If:
Difference is big,Noise is small=> p-value small → probably real.

Wilcoxon Test:
In how many trials was Pattern bigger than Baseline?

Sign Test:
It ignores magnitude.
Only looks at direction:

Trial 1 → Pattern > Baseline? Yes/No
Trial 2 → Pattern > Baseline? Yes/No
If 14 out of 15 trials are bigger → suspiciously consistent → likely real.

If 8 out of 15 → probably random.

Effect Size
How strong is the difference?
p-value tells you if it's real.
Effect size tells you how big it is.

Because averaging reduces noise as:

N∝(Effect Size1​)2
	​


But signal stays constant.

So to compensate for weak signal, you must aggressively reduce noise via averaging.