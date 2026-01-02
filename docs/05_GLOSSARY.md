# Glossary

Domain terms, abbreviations, and acronyms used in didactic-engine.

---

## Audio Terms

### Amplitude
The magnitude of an audio signal at a point in time. Measured in arbitrary units
for digital audio, typically normalized to [-1.0, 1.0].

### Beat
A single pulse in a musical rhythm. The time between beats determines tempo.

### Bar (Measure)
A segment of music containing a fixed number of beats, determined by the time
signature. In 4/4 time, one bar = 4 beats.

### BPM (Beats Per Minute)
A measure of tempo. 120 BPM means 120 beats occur every minute, or one beat
every 0.5 seconds.

### Chroma (Pitch Class)
A representation of audio that maps all pitches to the 12 semitones of the
chromatic scale (C, C#, D, ..., B). Octave-independent.

### dBFS (Decibels Full Scale)
A logarithmic measure of signal level where 0 dBFS is the maximum digital level.
-40 dBFS is a common silence threshold.

### FFT (Fast Fourier Transform)
An algorithm to convert audio from time domain to frequency domain. Used to
compute spectral features.

### Frame
A short segment of audio (typically 20-50ms) used for analysis. Audio is
processed as overlapping frames using a hop length.

### Hop Length
The number of samples between consecutive analysis frames. Smaller hop = more
frames = finer time resolution.

### MFCC (Mel-Frequency Cepstral Coefficients)
A compact representation of the spectral envelope, commonly used for audio
classification. Typically 13 coefficients are extracted.

### Mono
Single-channel audio. Stereo audio has two channels (left, right).

### Normalization
Scaling audio amplitude to a standard range, typically so the peak reaches
a target level (e.g., -1 dBFS).

### Sample Rate (SR)
The number of audio samples per second. Common rates: 22050 Hz (analysis),
44100 Hz (CD quality), 48000 Hz (video).

### Spectrogram
A visual representation of the frequency content of audio over time.
Computed via STFT.

### Spectral Centroid
The "center of mass" of a spectrum. High centroid = brighter sound.

### Spectral Rolloff
The frequency below which a specified percentage (typically 85%) of the
spectral energy is contained.

### Stem
An isolated component of a mix, such as vocals, drums, bass, or "other."
Created via source separation (e.g., Demucs).

### STFT (Short-Time Fourier Transform)
A Fourier transform applied to overlapping windows of audio to analyze
frequency content over time.

### Time Signature
The rhythmic structure of music, expressed as beats per bar / beat unit.
- 4/4: 4 quarter-note beats per bar (most common)
- 3/4: 3 quarter-note beats per bar (waltz)
- 6/8: 6 eighth-note beats per bar

### WAV
An uncompressed audio file format. Stores raw PCM samples.

### ZCR (Zero Crossing Rate)
The rate at which the audio signal changes sign. Higher ZCR often indicates
noisier or more percussive content.

---

## MIDI Terms

### MIDI (Musical Instrument Digital Interface)
A protocol and file format for representing musical events (notes, controls)
rather than audio samples.

### Note Event
A MIDI message representing a note being played. Contains:
- Pitch (0-127, where 60 = middle C)
- Velocity (0-127, loudness/intensity)
- Start time
- Duration

### Pitch
The perceived frequency of a note. In MIDI, represented as an integer 0-127.
- 60 = C4 (middle C)
- 69 = A4 (440 Hz)

### Program (Patch)
The instrument sound assigned to a MIDI track. 0-127 in General MIDI.

### Velocity
The intensity/loudness of a MIDI note. 0-127, where higher = louder.

---

## Data Terms

### DataFrame
A tabular data structure (from pandas) with rows and named columns.
Used for note events, beats, bars, and features.

### Parquet
A columnar file format for efficient storage and querying of tabular data.
Used for dataset output.

---

## Library Terms

### Basic Pitch
Spotify's audio-to-MIDI transcription library. Converts audio to MIDI notes.
https://github.com/spotify/basic-pitch

### Demucs
Facebook/Meta's source separation model. Separates audio into stems
(vocals, drums, bass, other). https://github.com/facebookresearch/demucs

### Essentia
An open-source library for audio analysis and MIR (Music Information Retrieval).
Provides advanced features beyond librosa. https://essentia.upf.edu/

### librosa
A Python library for audio and music analysis. Provides feature extraction,
beat tracking, and more. https://librosa.org/

### music21
A toolkit for computer-aided musicology. Used for ABC notation export.
https://web.mit.edu/music21/

### pretty_midi
A library for parsing and creating MIDI files. Provides note-level access
to MIDI data. https://github.com/craffel/pretty-midi

### pydub
A simple audio manipulation library. Used for normalization, trimming,
and slicing. https://github.com/jiaaro/pydub

### soundfile
A library for reading and writing audio files. Used for WAV I/O.
https://python-soundfile.readthedocs.io/

---

## Pipeline Terms

### Bar Chunk
A segment of audio corresponding to a single bar (measure). Written as
individual WAV files for per-bar analysis.

### Bar Features
Audio features extracted from individual bar chunks. Enables analysis of
how features evolve across a song.

### Beat Grid
A list of beat times (in seconds) detected from audio. Used for alignment
and bar computation.

### Pipeline
The complete processing workflow from input WAV to output datasets and reports.
Consists of 14 sequential steps.

### Song ID
A unique identifier for a song being processed. Used to organize output
directories.

---

## Abbreviations

| Abbrev | Meaning |
|--------|---------|
| ABC | A text-based music notation format |
| BPM | Beats Per Minute |
| CLI | Command Line Interface |
| dBFS | Decibels Full Scale |
| FFT | Fast Fourier Transform |
| MFCC | Mel-Frequency Cepstral Coefficients |
| MIDI | Musical Instrument Digital Interface |
| MIR | Music Information Retrieval |
| PCM | Pulse Code Modulation |
| RMS | Root Mean Square |
| SR | Sample Rate |
| STFT | Short-Time Fourier Transform |
| WAV | Waveform Audio File Format |
| ZCR | Zero Crossing Rate |

---

## See Also

- [Architecture](01_ARCHITECTURE.md) - System design
- [Key Flows](02_KEY_FLOWS.md) - Pipeline execution
