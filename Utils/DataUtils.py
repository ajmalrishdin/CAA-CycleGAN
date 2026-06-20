import os

import pyedflib
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale
import padasip as pa
import wfdb
from Utils.utils import denorm





class DataUtils:

    def __init__(self) -> None:
        super().__init__()
        # Paper setup: use 60 seconds starting from the 30th second.
        self.original_fs = 1000
        self.segment_start_sec = 30
        self.segment_duration_sec = 60
        self.dataPath = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "Databases",
                "ADFECGDB",
            )
        )
        self.fileNames = self._load_file_names(self.dataPath)
        self._print_training_files()

    def _load_file_names(self, path):
        return sorted(
            file_name
            for file_name in os.listdir(path)
            if file_name.lower().endswith(".edf")
            and os.path.isfile(os.path.join(path, file_name))
        )

    def _print_training_files(self):
        print(f"Training folder: {self.dataPath}")
        print(self.fileNames)

    def readData(self, sigNum, path=None):
        if path is None:
            path = self.dataPath

        if not self.fileNames or os.path.abspath(path) != self.dataPath:
            self.fileNames = self._load_file_names(path)

        file_name = os.path.join(path, self.fileNames[sigNum])
        f = pyedflib.EdfReader(file_name)
        n = f.signals_in_file
        start_sample = int(self.segment_start_sec * self.original_fs)
        end_sample = int((self.segment_start_sec + self.segment_duration_sec) * self.original_fs)
        signal_len = f.getNSamples()[0]
        start_sample = max(0, min(start_sample, signal_len))
        end_sample = max(start_sample, min(end_sample, signal_len))

        # signal_labels = f.getSignalLabels()
        abdECG = np.zeros((n - 1, end_sample - start_sample))
        fetalECG = np.zeros((1, end_sample - start_sample))
        fetalECG[0, :] = f.readSignal(0)[start_sample:end_sample]
        fetalECG[0, :] = scale(self.SignalFilter(self.butter_bandpass_filter(fetalECG, 1, 100, 1000)), axis=1)
        for i in np.arange(1, n):
            abdECG[i - 1, :] = f.readSignal(i)[start_sample:end_sample]
        abdECG = scale(self.SignalFilter(self.butter_bandpass_filter(abdECG, 1, 100, 1000)), axis=1)


        # abdECG = self.normalize(signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1))
        # fetalECG = self.normalize(signal.resample(fetalECG, int(fetalECG.shape[1] / 5), axis=1))
        
        abdECG = signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1)
        fetalECG = signal.resample(fetalECG, int(fetalECG.shape[1] / 5), axis=1)
        
        

        signal_annotation = wfdb.rdann(file_name, "qrs", sampfrom=start_sample, sampto=end_sample)
        fqrs_rpeaks = signal_annotation.sample
        fqrs_rpeaks = fqrs_rpeaks[(fqrs_rpeaks >= start_sample) & (fqrs_rpeaks < end_sample)]
        fqrs_rpeaks = np.asarray(np.floor_divide(fqrs_rpeaks - start_sample, 5), 'int64')
        
        return abdECG, fetalECG,fqrs_rpeaks

    def windowingSig(self, sig1, sig2, fqrs_rpeaks, windowSize=15):
        signalLen = sig2.shape[1]
        signalsWindow1 = [sig1[:, int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]
        signalsWindow2 = [sig2[:, int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]

        # fqrsWindows = np.zeros([len(signalsWindow2),1]).astype(int)
        fqrsWindows = []
        ik = 0
        # handle case where no detected peaks are available
        n_peaks = 0 if fqrs_rpeaks is None else fqrs_rpeaks.size
        if n_peaks == 0:
            # return an empty list of peaks for each window
            fqrsWindows = [[] for _ in range(len(signalsWindow2))]
            return signalsWindow1, signalsWindow2, fqrsWindows

        for i in range(0, signalLen - windowSize, windowSize):
            fqrs = []
            # advance through peaks that fall before the end of this window
            while ik < n_peaks and (fqrs_rpeaks[ik] < int(i + windowSize)):
                index = fqrs_rpeaks[ik]
                if int(i) <= index < int(i + windowSize):
                    fqrs.append(index - int(i))
                ik += 1
            fqrsWindows.append(fqrs)
                
        
        return signalsWindow1, signalsWindow2, fqrsWindows

    def adaptFilterOnSig(self, src, ref):
        f = pa.filters.FilterNLMS(n=4, mu=0.1, w="random")
        for index, sig in enumerate(src):
            try:
                y, e, w = f.run(ref[index][:, 0], sig)
                ref[index][:, 0] = e
            except:
                pass

        return ref

    def calculateICA(self, sdSig, component=7):
        ica = FastICA(n_components=component, max_iter=1000)
        icaRes = []
        for index, sig in enumerate(sdSig):
            try:
                icaSignal = np.array(ica.fit_transform(sig))
                icaSignal = np.append(icaSignal, sig[:, range(2, 4)], axis=1)
                icaRes.append(icaSignal)
            except:
                pass
        return np.array(icaRes)

    def createDelayRepetition(self, signal, numberDelay=4, delay=10):
        signal = np.repeat(signal, numberDelay, axis=0)
        for row in range(1, signal.shape[0]):
            signal[row, :] = np.roll(signal[row, :], shift=delay * row)
        return signal

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, axis=1):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=axis)
        return y
    
    def SignalFilter(self,filtedData):
        A = np.array([1,0,0,0,0,0,0,0,0,0,-0.854]); #梳状滤波器系数
        B = np.array([0.927,0,0,0,0,0,0,0,0,0,-0.927]);
        
        filtedData = signal.filtfilt(B, A, filtedData)
        
        B1 = np.array([0.995,-1.8504,0.995]);# 陷波滤波器系数
        A1 = np.array([1,-1.8505,0.99]);
        filtedData = signal.filtfilt(B1, A1, filtedData)
    
        B2 = np.array([0.388,0.388]); #Wc = 180，3dB的截止频率
        A2 = np.array([1,-0.42578]);
        filtedData = signal.filtfilt(B2, A2, filtedData)
        return filtedData
    
    
    def normalize(self,v):
        v_min = v.min(axis=1).reshape((v.shape[0],1))
        v_max = v.max(axis=1).reshape((v.shape[0],1))
        return (v - v_min) / (v_max-v_min)

