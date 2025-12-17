from dataclasses import dataclass
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import List, Optional

@dataclass(frozen=True)
class TranscriptLine:
    text: str
    start: float
    duration: float

class YouTubeTranscriptLoader:
    """Loads a transcript from YouTube (manual captions preferred, fallback to generated)."""

    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = languages or ["en"]

    def load(self, video_id: str) -> List[TranscriptLine]:
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)

            # Prefer manually created transcript if available
            transcript = None
            try:
                transcript = transcript_list.find_manually_created_transcript(self.languages)
            except Exception:
                transcript = transcript_list.find_generated_transcript(self.languages)

            items = transcript.fetch()
            lines: List[TranscriptLine] = []
            for it in items:
                text = (it.get("text") or "").replace("\n", " ").strip()
                start = float(it.get("start") or 0.0)
                duration = float(it.get("duration") or 0.0)
                if text:
                    lines.append(TranscriptLine(text=text, start=start, duration=duration))
            return lines

        except (TranscriptsDisabled, NoTranscriptFound):
            raise RuntimeError("No transcript available for this video (disabled or not found).")
        except Exception as e:
            raise RuntimeError(f"Failed to load transcript: {e}") from e    

