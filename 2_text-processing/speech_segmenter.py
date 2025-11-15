"""
Speech Segmentation Module

Segments transcript into meaningful chunks (sentences/phrases) while preserving
word-level timing information for synchronization with sound effects.
"""

from typing import List, Dict, Tuple
import re


class SpeechSegmenter:
    """Segments speech transcript into chunks with timing information."""

    def __init__(self, max_words_per_segment: int = 15):
        """
        Initialize the segmenter.

        Args:
            max_words_per_segment: Maximum number of words per segment.
                                   Prevents segments from being too long for embedding.
        """
        self.max_words_per_segment = max_words_per_segment

    def segment_by_sentences(
        self,
        transcript: str,
        word_timings: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Segment transcript into sentences while preserving timing information.

        Args:
            transcript: Full transcript text
            word_timings: List of word timing dictionaries with structure:
                         [{'word': str, 'start_time': float, 'end_time': float}, ...]

        Returns:
            List of segment dictionaries with structure:
            [
                {
                    'segment_id': int,
                    'text': str,
                    'start_time': float,
                    'end_time': float,
                    'word_count': int,
                    'words': List[Dict]  # Original word timings for this segment
                },
                ...
            ]
        """
        # Split transcript into sentences using basic punctuation
        sentence_pattern = r'[.!?]+\s*'
        sentences = re.split(sentence_pattern, transcript)
        sentences = [s.strip() for s in sentences if s.strip()]

        segments = []
        word_index = 0

        for sent_idx, sentence in enumerate(sentences):
            # Split long sentences into smaller chunks
            sentence_words = sentence.split()

            if len(sentence_words) > self.max_words_per_segment:
                # Break into smaller chunks
                for i in range(0, len(sentence_words), self.max_words_per_segment):
                    chunk_words = sentence_words[i:i + self.max_words_per_segment]
                    chunk_text = ' '.join(chunk_words)

                    segment = self._create_segment(
                        segment_id=len(segments),
                        text=chunk_text,
                        word_timings=word_timings,
                        word_index=word_index,
                        word_count=len(chunk_words)
                    )

                    if segment:
                        segments.append(segment)
                    word_index += len(chunk_words)
            else:
                # Use entire sentence as segment
                segment = self._create_segment(
                    segment_id=len(segments),
                    text=sentence,
                    word_timings=word_timings,
                    word_index=word_index,
                    word_count=len(sentence_words)
                )

                if segment:
                    segments.append(segment)
                word_index += len(sentence_words)

        return segments

    def _create_segment(
        self,
        segment_id: int,
        text: str,
        word_timings: List[Dict],
        word_index: int,
        word_count: int
    ) -> Dict[str, any]:
        """
        Create a segment with timing information.

        Args:
            segment_id: Unique identifier for this segment
            text: The text content of the segment
            word_timings: Full list of word timings
            word_index: Starting index in word_timings
            word_count: Number of words in this segment

        Returns:
            Segment dictionary or None if timing information is unavailable
        """
        # Extract word timings for this segment
        end_index = min(word_index + word_count, len(word_timings))

        if word_index >= len(word_timings):
            # No timing information available for this segment
            return None

        segment_words = word_timings[word_index:end_index]

        if not segment_words:
            return None

        # Calculate segment start and end times
        start_time = segment_words[0]['start_time']
        end_time = segment_words[-1]['end_time']

        return {
            'segment_id': segment_id,
            'text': text,
            'start_time': start_time,
            'end_time': end_time,
            'word_count': len(segment_words),
            'words': segment_words  # Preserve word-level detail
        }

    def segment_by_time_windows(
        self,
        transcript: str,
        word_timings: List[Dict[str, any]],
        window_seconds: float = 5.0,
        overlap_seconds: float = 0.0
    ) -> List[Dict[str, any]]:
        """
        Segment transcript using fixed time windows (alternative approach).

        Args:
            transcript: Full transcript text
            word_timings: List of word timing dictionaries
            window_seconds: Duration of each time window in seconds
            overlap_seconds: Overlap between consecutive windows

        Returns:
            List of segment dictionaries (same structure as segment_by_sentences)
        """
        if not word_timings:
            return []

        segments = []
        current_window_start = word_timings[0]['start_time']
        total_duration = word_timings[-1]['end_time']
        segment_id = 0

        while current_window_start < total_duration:
            window_end = current_window_start + window_seconds

            # Find words within this time window
            window_words = [
                w for w in word_timings
                if w['start_time'] >= current_window_start and w['end_time'] <= window_end
            ]

            if window_words:
                text = ' '.join([w['word'] for w in window_words])

                segment = {
                    'segment_id': segment_id,
                    'text': text,
                    'start_time': window_words[0]['start_time'],
                    'end_time': window_words[-1]['end_time'],
                    'word_count': len(window_words),
                    'words': window_words
                }

                segments.append(segment)
                segment_id += 1

            # Move to next window (with optional overlap)
            current_window_start += (window_seconds - overlap_seconds)

        return segments


def load_stt_output(stt_result: Dict) -> Tuple[str, List[Dict]]:
    """
    Extract transcript and word timings from Google Speech API output.

    Args:
        stt_result: Output from stt_google.transcribe_audio()

    Returns:
        Tuple of (transcript, word_timings)
    """
    transcript = stt_result['results']['transcript']
    word_timings = stt_result['words_timings']

    return transcript, word_timings
