
import torchaudio


class SPM:
    def process(waveform, sample_rate):
        """
        提取梅尔频谱图
        Args:
            waveform (Tensor): 波形数据
            sample_rate (int): 采样率
        Returns:
            mel_spectrogram (Tensor): 梅尔频谱图
        """
        # 提取梅尔频谱图
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )(waveform)
        
        # 转换为分贝单位
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        return mel_spectrogram