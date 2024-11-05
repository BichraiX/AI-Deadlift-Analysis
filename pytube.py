import youtube_dl

def download_video(url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': '%(title)s.%(ext)s',  # Save with the video title as filename
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

video_url = "https://www.youtube.com/watch?v=AweC3UaM14o"
download_video(video_url)