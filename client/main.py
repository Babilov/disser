import asyncio
from VideoProccessor import VideoProcessor


if __name__ == '__main__':
    FPS = 30
    DELAY = 1 / FPS
    VIDEO_PATH = 'client/videos/road3.mp4'
    WINDOW_NAME = 'Road'
    SERVER_URL = "ws://127.0.0.1:8000/video"  # Используем WebSocket URL
    
    try:
        video_processor = VideoProcessor(VIDEO_PATH, WINDOW_NAME, SERVER_URL, DELAY)
        asyncio.run(video_processor.process())
    except Exception as e:
        print(e)