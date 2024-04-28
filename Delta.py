#!/usr/bin/env python
# coding: utf-8

# Experiment 3: Delta Modulation

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

#### CASE 1:  fs = 40*f;    delta = (2*pi*A*f)/fs = 0.785398

A = 5
f = 100
T = 2 / f
fs = 40 * f
t = np.arange(0, T, 1 / fs)
x = A * np.sin(2 * np.pi * f * t)

# Plotting the original sine wave
plt.figure(figsize=(20, 4))
plt.plot(t, x)
plt.title('Original Sinusoidal Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Delta modulation
delta = 0.78539
xn_mod = np.zeros(len(x) + 1)  # Extend xn_mod by one element
d = np.zeros(len(x), dtype=int)

for i in range(len(x)):
    if x[i] > xn_mod[i]:
        d[i] = 1
        xn_mod[i + 1] = xn_mod[i] + delta
    else:
        d[i] = 0
        xn_mod[i + 1] = xn_mod[i] - delta
        
# Plotting both signals on the same graph
plt.figure(figsize=(20, 6))

# Plot the original sinusoidal signal
plt.plot(t, x, label='Original Signal')

# Plot the delta modulated signal
plt.step(np.arange(len(xn_mod) - 1) * T / len(x), xn_mod[:-1], label='Delta Modulated Signal')

plt.title('Comparison of Original Signal and Delta Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Encoding signal
plt.figure(figsize=(20, 6))
plt.step(np.arange(len(d)) * T / len(x), d, label='Encoded Signal', linestyle='--')
plt.title('Encoded Signal')
plt.xlabel('Time (s)')
plt.ylabel('Encoded Value')
plt.grid(True)
plt.legend()
plt.show()

# Plotting delta modulated signal
plt.figure(figsize=(20, 6))
plt.step(np.arange(len(xn_mod) - 1) * T / len(x), xn_mod[:-1], label='Decoded Signal')
plt.title('Decoded Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Demodulation
demodulated_signal = np.cumsum(xn_mod[1:])  # Demodulation
t_demod = np.linspace(0, T, len(demodulated_signal))  # Adjusting time scale

# Plotting the demodulated signal
plt.figure(figsize=(20, 4))
plt.plot(t_demod, demodulated_signal)
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

#### CASE 2:  fs = 40*f;    delta < (2*pi*A*f)/fs so delta = 0.55

A = 5
f = 100
T = 2 / f
fs = 40 * f
t = np.arange(0, T, 1 / fs)
x = A * np.sin(2 * np.pi * f * t)

# Plotting the original sine wave
plt.figure(figsize=(20, 4))
plt.plot(t, x)
plt.title('Original Sinusoidal Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Delta modulation
delta = 0.55
xn_mod = np.zeros(len(x) + 1)  # Extend xn_mod by one element
d = np.zeros(len(x), dtype=int)

for i in range(len(x)):
    if x[i] > xn_mod[i]:
        d[i] = 1
        xn_mod[i + 1] = xn_mod[i] + delta
    else:
        d[i] = 0
        xn_mod[i + 1] = xn_mod[i] - delta
        
# Plotting both signals on the same graph
plt.figure(figsize=(20, 6))

# Plot the original sinusoidal signal
plt.plot(t, x, label='Original Signal')

# Plot the delta modulated signal
plt.step(np.arange(len(xn_mod) - 1) * T / len(x), xn_mod[:-1], label='Delta Modulated Signal')

plt.title('Comparison of Original Signal and Delta Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Encoding signal
plt.figure(figsize=(20, 6))
plt.step(np.arange(len(d)) * T / len(x), d, label='Encoded Signal', linestyle='--')
plt.title('Encoded Signal')
plt.xlabel('Time (s)')
plt.ylabel('Encoded Value')
plt.grid(True)
plt.legend()
plt.show()

# Plotting delta modulated signal
plt.figure(figsize=(20, 6))
plt.step(np.arange(len(xn_mod) - 1) * T / len(x), xn_mod[:-1], label='Decoded Signal')
plt.title('Decoded Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Demodulation
demodulated_signal = np.cumsum(xn_mod[1:])  # Demodulation
t_demod = np.linspace(0, T, len(demodulated_signal))  # Adjusting time scale

# Plotting the demodulated signal
plt.figure(figsize=(20, 4))
plt.plot(t_demod, demodulated_signal)
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

#### CASE 3:  fs = 40*f;    delta > (2*pi*A*f)/fs so delta = 1.5

A = 5
f = 100
T = 2 / f
fs = 40 * f
t = np.arange(0, T, 1 / fs)
x = A * np.sin(2 * np.pi * f * t)

# Plotting the original sine wave
plt.figure(figsize=(20, 4))
plt.plot(t, x)
plt.title('Original Sinusoidal Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Delta modulation
delta = 1.5
xn_mod = np.zeros(len(x) + 1)  # Extend xn_mod by one element
d = np.zeros(len(x), dtype=int)

for i in range(len(x)):
    if x[i] > xn_mod[i]:
        d[i] = 1
        xn_mod[i + 1] = xn_mod[i] + delta
    else:
        d[i] = 0
        xn_mod[i + 1] = xn_mod[i] - delta
        
# Plotting both signals on the same graph
plt.figure(figsize=(20, 6))

# Plot the original sinusoidal signal
plt.plot(t, x, label='Original Signal')

# Plot the delta modulated signal
plt.step(np.arange(len(xn_mod) - 1) * T / len(x), xn_mod[:-1], label='Delta Modulated Signal')

plt.title('Comparison of Original Signal and Delta Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Encoding signal
plt.figure(figsize=(20, 6))
plt.step(np.arange(len(d)) * T / len(x), d, label='Encoded Signal', linestyle='--')
plt.title('Encoded Signal')
plt.xlabel('Time (s)')
plt.ylabel('Encoded Value')
plt.grid(True)
plt.legend()
plt.show()

# Plotting delta modulated signal
plt.figure(figsize=(20, 6))
plt.step(np.arange(len(xn_mod) - 1) * T / len(x), xn_mod[:-1], label='Decoded Signal')
plt.title('Decoded Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Demodulation
demodulated_signal = np.cumsum(xn_mod[1:])  # Demodulation
t_demod = np.linspace(0, T, len(demodulated_signal))  # Adjusting time scale

# Plotting the demodulated signal
plt.figure(figsize=(20, 4))
plt.plot(t_demod, demodulated_signal)
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


# ADAPTIVE DELTA MODULATION:

# In[5]:


A = 10
f = 50
T = 2 / f
fs = 20 * f
t = np.arange(0, T, 1 / fs)
x = A * np.sin(2 * np.pi * f * t)

# Plotting the original sine wave
plt.figure(figsize=(20, 4))
plt.plot(t, x)
plt.title('Original Sinusoidal Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




