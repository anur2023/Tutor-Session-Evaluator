import subprocess

def download_and_convert_m3u8(m3u8_url, output_file):
    try:
        command = [
            "ffmpeg",
            "-i", m3u8_url,
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-b:a", "192k",  # Audio bitrate
            output_file
        ]
        
        subprocess.run(command, check=True)
        print(f"Conversion completed: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Error during conversion:", e)
