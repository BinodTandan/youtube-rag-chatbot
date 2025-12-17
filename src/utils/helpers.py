import re

def extract_video_id(url: str) -> str:
    "Extract YouTube video id from URL"

    if not url or not isinstance(url, str):
        return ValueError("Invalid URL")
    
    patterns = [
        r"youtu\.be/([^?\s/]+)",
        r"youtube\.com/watch\?v=([^&\s]+)",
        r"youtube\.com/shorts/([^?\s/]+)",
        r"youtube\.com/embed/([^?\s/]+)",
        r"v=([^&\s]+)"
    ]

    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
        
    raise ValueError("Unable to extract video ID from the given URL")


    def format_time(seconds: float) -> str:
        seconds = float(seconds or 0.0)
        m = int(seconds // 60)
        s = int(seconds % 60)

        return f"{m:02d}:{s:02d}"
    

    def youtube_timestamp_url(video_id: str, start_seconds: float) -> str:
        return f"https://www.youtube.com/watch?v={video_id}&t={int(float(start_seconds))}s"



