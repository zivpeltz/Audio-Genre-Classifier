from __future__ import unicode_literals
import youtube_dl
link = r"https://www.youtube.com/playlist?list=PLNxOe-buLm6cz8UQ-hyG1nm3RTNBUBv3K"

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320',
    }],
}


with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([link])