from pydub import AudioSegment
import os

def extract_audio_from_video(video_path, audio_output_path):
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_output_path, format="wav")
        return True
    except Exception as e:
        print(f"❌ Failed to extract audio: {e}")
        return False
