"""Microphone input for voice control, as 16 kHz mono float32 blocks, plus streaming speech detection.

Two inputs:
- `AppleVoiceInput`: Apple's voice processing (the same as Voice Isolation / FaceTime): noise suppression, echo
  cancellation (the Mac's own audio is removed from the mic) and level control. It only works on the system
  default input; picking another device makes it fail to initialise (measured: AVAudioEngine error -10875).
- `PortAudioInput`: the raw stream from any device, for a mic chosen by name.
`SpeechDetector` is Silero VAD run one 32 ms chunk at a time (~0.1 ms each) so only speech starts an utterance.
"""
from __future__ import annotations

import os
import queue
import warnings

import numpy as np

RATE = 16000
CHUNK = 512  # Silero's frame at 16 kHz: 32 ms
DUCK_MIN = 10  # AVAudioVoiceProcessingOtherAudioDuckingLevelMin: turn other apps down as little as possible


class Resampler:
    """Streaming band-limited resampler (windowed sinc), any rate ratio: ~0.5 ms per 100 ms block at 24 kHz."""

    def __init__(self, rate_in: float, rate_out: float = RATE, half_taps: int = 16):
        self.step = rate_in / rate_out  # input samples per output sample
        self.cut = min(1.0, rate_out / rate_in) * 0.95  # low-pass below the output's Nyquist: no aliasing
        self.w = half_taps
        self.offs = np.arange(-half_taps + 1, half_taps + 1)
        self.buf = np.zeros(2 * half_taps, np.float32)  # input history the next window needs
        self.pos = float(half_taps)  # next output position, in `buf` samples

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.step == 1.0:
            return np.asarray(x, np.float32)
        buf = np.concatenate([self.buf, np.asarray(x, np.float32)])
        last = len(buf) - 1 - self.w  # latest position with a full window of input
        n = int(np.floor((last - self.pos) / self.step)) + 1 if last >= self.pos else 0
        p = self.pos + self.step * np.arange(n)
        i0 = np.floor(p).astype(np.int64)
        t = self.offs[None, :] - (p - i0)[:, None]
        h = self.cut * np.sinc(self.cut * t) * (0.5 + 0.5 * np.cos(np.pi * t / self.w))  # Hann-windowed sinc
        y = (buf[i0[:, None] + self.offs[None, :]] * h).sum(1).astype(np.float32)
        nxt = self.pos + self.step * n
        keep = max(int(np.floor(nxt)) - self.w + 1, 0)
        self.buf, self.pos = buf[keep:], nxt - keep
        return y


class SpeechDetector:
    """Silero VAD (shipped with faster-whisper) with its recurrent state kept between 32 ms chunks."""

    def __init__(self):
        import onnxruntime as ort
        from faster_whisper.vad import get_assets_path

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = ort.InferenceSession(os.path.join(get_assets_path(), "silero_vad_v6.onnx"),
                                             providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros((1, 64), np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """Probability that this CHUNK-sample chunk is speech."""
        x = np.asarray(chunk, np.float32)[None, :]
        out, self._h, self._c = self._session.run(
            None, {"input": np.concatenate([self._context, x], 1), "h": self._h, "c": self._c})
        self._context = x[:, -64:]
        return float(out.ravel()[0])


class _Input:
    """Pushes 16 kHz float32 blocks into `blocks` while started."""

    label = ""

    def __init__(self):
        self.blocks: queue.Queue = queue.Queue()

    def start(self) -> None: ...
    def stop(self) -> None: ...


class PortAudioInput(_Input):
    def __init__(self, device: int | None):
        super().__init__()
        self.device = device

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(device=self.device, samplerate=RATE, channels=1, dtype="float32",
                                      blocksize=CHUNK, callback=lambda data, *_: self.blocks.put(data[:, 0].copy()))
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


class AppleVoiceInput(_Input):
    """The system default input through Apple's voice processing (AVAudioEngine, voice processing I/O)."""

    label = "voice isolation"

    def start(self) -> None:
        import AVFoundation as AV

        self._engine = AV.AVAudioEngine.alloc().init()
        node = self._engine.inputNode()
        ok, err = node.setVoiceProcessingEnabled_error_(True, None)
        if not ok:
            raise RuntimeError(f"voice processing unavailable: {err}")
        try:  # macOS 14+: don't turn the user's music down more than necessary
            node.setVoiceProcessingOtherAudioDuckingConfiguration_(
                AV.AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking=False,
                                                                        duckingLevel=DUCK_MIN))
        except Exception:
            pass
        fmt = node.outputFormatForBus_(0)  # e.g. 24 kHz, 3 channels; channel 0 is the processed voice
        resample = Resampler(fmt.sampleRate())

        def tap(buffer, _when):
            n = buffer.frameLength()
            samples = np.frombuffer(buffer.floatChannelData()[0].as_buffer(n), np.float32)
            self.blocks.put(resample(samples))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # PyObjC warns about the raw AudioUnit pointer type
            node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, tap)
        self._engine.prepare()
        ok, err = self._engine.startAndReturnError_(None)
        if not ok:
            node.removeTapOnBus_(0)
            raise RuntimeError(f"voice processing failed to start: {err}")

    def stop(self) -> None:
        self._engine.inputNode().removeTapOnBus_(0)
        self._engine.stop()
