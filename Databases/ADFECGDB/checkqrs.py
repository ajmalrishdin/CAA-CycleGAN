import pyedflib, pandas as pd, matplotlib.pyplot as plt, numpy as np

f = pyedflib.EdfReader(r'C:\Users\obsid\Downloads\Synthetic Database\r07_ARR_4.edf')
sig = f.readSignal(0)
fs = f.getSampleFrequency(0)
f.close()

df = pd.read_csv(r'C:\Users\obsid\Downloads\Synthetic Database\r07_ARR_4.qrs')
t = np.arange(len(sig)) / fs

plt.plot(t, sig, lw=0.7)
plt.scatter(df.time_sec, sig[df.sample_index], color='red', s=20, zorder=5)
#plt.xlim(175, 195)
plt.title('Zoom 175-195s')
plt.xlabel('Time (s)')
plt.show()