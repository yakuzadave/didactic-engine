"""
Main audio processing pipeline.

Orchestrates the complete audio processing workflow from ingestion to feature extraction.
"""

import os
from typing import Dict, List, Optional, Any
import numpy as np
import json

from didactic_engine.ingestion import WAVIngester
from didactic_engine.separation import StemSeparator
from didactic_engine.preprocessing import AudioPreprocessor
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.transcription import MIDITranscriber
from didactic_engine.midi_parser import MIDIParser
from didactic_engine.segmentation import StemSegmenter
from didactic_engine.features import FeatureExtractor


class AudioPipeline:
    """
    Complete audio processing pipeline.
    
    Ingests WAV audio, separates stems, analyzes audio, transcribes to MIDI,
    aligns to beat/bar grid, segments stems, and extracts features.
    """

    def __init__(
        self,
        sample_rate: Optional[int] = 44100,
        use_essentia: bool = False,
        preprocess_stems: bool = True,
        beats_per_bar: int = 4,
    ):
        """
        Initialize the audio pipeline.

        Args:
            sample_rate: Target sample rate for processing.
            use_essentia: Whether to use Essentia for additional analysis.
            preprocess_stems: Whether to preprocess stems after separation.
            beats_per_bar: Number of beats per bar for bar detection.
        """
        self.sample_rate = sample_rate
        self.use_essentia = use_essentia
        self.preprocess_stems = preprocess_stems
        self.beats_per_bar = beats_per_bar

        # Initialize components
        self.ingester = WAVIngester(sample_rate=sample_rate)
        self.separator = StemSeparator()
        self.preprocessor = AudioPreprocessor()
        self.analyzer = AudioAnalyzer(use_essentia=use_essentia)
        self.transcriber = MIDITranscriber()
        self.midi_parser = MIDIParser()
        self.segmenter = StemSegmenter()
        self.feature_extractor = FeatureExtractor()

    def process(
        self, input_wav_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """
        Run the complete audio processing pipeline.

        Args:
            input_wav_path: Path to input WAV file.
            output_dir: Directory to save all outputs.

        Returns:
            Dictionary containing all processing results and output paths.
        """
        print(f"Starting audio processing pipeline for: {input_wav_path}")
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "input_path": input_wav_path,
            "output_dir": output_dir,
        }

        # Step 1: Ingest WAV
        print("Step 1: Ingesting WAV file...")
        audio, sr = self.ingester.load(input_wav_path)
        results["sample_rate"] = sr
        results["audio_shape"] = audio.shape
        print(f"  Loaded audio: shape={audio.shape}, sr={sr}")

        # Step 2: Analyze audio
        print("Step 2: Analyzing audio...")
        analysis = self.analyzer.analyze(audio, sr)
        results["analysis"] = analysis
        print(f"  Detected tempo: {analysis['tempo']:.2f} BPM")
        print(f"  Found {len(analysis['beat_times'])} beats")

        # Extract beat and bar times
        beat_times = np.array(analysis["beat_times"])
        bar_times = self.analyzer.extract_bar_times(audio, sr, self.beats_per_bar)
        results["beat_times"] = beat_times.tolist()
        results["bar_times"] = bar_times.tolist()
        print(f"  Extracted {len(bar_times)} bars")

        # Step 3: Separate stems
        print("Step 3: Separating stems...")
        stems_dir = os.path.join(output_dir, "stems")
        stems = self.separator.separate(audio, sr, stems_dir)
        results["stem_names"] = list(stems.keys())
        print(f"  Separated into {len(stems)} stems: {list(stems.keys())}")

        # Step 4: Preprocess stems (optional)
        if self.preprocess_stems:
            print("Step 4: Preprocessing stems...")
            preprocessed_stems = {}
            for stem_name, stem_audio in stems.items():
                try:
                    preprocessed = self.preprocessor.normalize(stem_audio, sr)
                    preprocessed_stems[stem_name] = preprocessed
                    print(f"  Preprocessed stem: {stem_name}")
                except Exception as e:
                    print(f"  Warning: Could not preprocess {stem_name}: {e}")
                    preprocessed_stems[stem_name] = stem_audio
            stems = preprocessed_stems

        # Step 5: Transcribe to MIDI
        print("Step 5: Transcribing to MIDI...")
        try:
            model_output, midi_data, note_events = self.transcriber.transcribe(audio, sr)
            midi_path = os.path.join(output_dir, "transcription.mid")
            self.transcriber.save_midi(midi_data, midi_path)
            results["midi_path"] = midi_path
            print(f"  Transcribed and saved MIDI to: {midi_path}")

            # Step 6: Parse MIDI
            print("Step 6: Parsing MIDI...")
            midi_info = self.midi_parser.parse(midi_data)
            results["midi_info"] = midi_info
            print(f"  Found {midi_info['total_notes']} notes in MIDI")

            # Step 7: Align MIDI to beat/bar grid
            print("Step 7: Aligning MIDI to bar grid...")
            aligned_notes = self.midi_parser.align_to_grid(
                midi_data, bar_times, quantize=True
            )
            results["aligned_notes"] = {
                str(k): v for k, v in aligned_notes.items()
            }
            print(f"  Aligned notes across {len(aligned_notes)} bars")
        except Exception as e:
            print(f"  Warning: MIDI transcription failed: {e}")
            results["midi_error"] = str(e)

        # Step 8: Segment stems into per-bar chunks
        print("Step 8: Segmenting stems into per-bar chunks...")
        segments_dir = os.path.join(output_dir, "segments")
        segmented_stems = self.segmenter.segment_stems_by_bars(
            stems, sr, bar_times, segments_dir
        )
        results["segmented_stems"] = {
            stem: len(chunks) for stem, chunks in segmented_stems.items()
        }
        print(f"  Segmented {len(stems)} stems into per-bar chunks")

        # Step 9: Extract bar-level features
        print("Step 9: Extracting bar-level features...")
        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        
        all_stem_features = {}
        for stem_name, chunk_paths in segmented_stems.items():
            print(f"  Extracting features for stem: {stem_name}")
            features = self.feature_extractor.extract_features_from_chunks(
                chunk_paths, sr
            )
            all_stem_features[stem_name] = features
            
            # Save features to JSON
            features_path = os.path.join(features_dir, f"{stem_name}_features.json")
            with open(features_path, "w") as f:
                json.dump(features, f, indent=2)
        
        results["features"] = {
            stem: len(feats) for stem, feats in all_stem_features.items()
        }
        print(f"  Extracted features for {len(all_stem_features)} stems")

        # Save complete results
        results_path = os.path.join(output_dir, "pipeline_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPipeline completed! Results saved to: {results_path}")

        return results

    def process_batch(
        self, input_wav_paths: List[str], output_base_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Process multiple WAV files through the pipeline.

        Args:
            input_wav_paths: List of paths to input WAV files.
            output_base_dir: Base directory for outputs.

        Returns:
            List of result dictionaries, one per input file.
        """
        all_results = []
        
        for i, input_path in enumerate(input_wav_paths):
            print(f"\n{'='*60}")
            print(f"Processing file {i+1}/{len(input_wav_paths)}: {input_path}")
            print(f"{'='*60}")
            
            # Create output directory for this file
            basename = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.join(output_base_dir, basename)
            
            try:
                results = self.process(input_path, output_dir)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                all_results.append({
                    "input_path": input_path,
                    "error": str(e),
                })
        
        return all_results
