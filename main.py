from tools import FeatureExtractor
import librosa

def main():
    audio_path = 'samples/track_1.wav'
    x, sr = librosa.load(audio_path)
    FeatureExtractor.VisualizeWaveForm(x, sr)
    FeatureExtractor.VisualizeSpectrogram(x, sr)
    row = FeatureExtractor.Extract(x, sr)
    print(row)

if __name__ == '__main__':
    main()


