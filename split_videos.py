import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import subprocess, math, shlex

clip_duration = 5

source_file_dir = '/media/asif/Programming/Thesis/Video_Classifier/Filter_Video/Input/'
des_file_dir = '/media/asif/Programming/Thesis/Video_Classifier/Filter_Video/Output/'

def ceildiv(a, b):
    return int(math.ceil(a / float(b)))

def get_video_length(filename):
    output = subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                      "default=noprint_wrappers=1:nokey=1", filename)).strip()
    video_length = int(float(output))
    return video_length

def split_by_seconds(filename, split_length, des, vcodec="copy", acodec="copy",
                     extra="", video_length=None, **kwargs):
    if split_length and split_length <= 0:
        raise SystemExit
    if not video_length:
        video_length = get_video_length(filename)
    split_count = ceildiv(video_length, split_length)
    if split_count == 1:
        raise SystemExit
    split_cmd = ["ffmpeg", "-i", filename, "-vcodec", vcodec, "-acodec", acodec] + shlex.split(extra)
    try:
        filebase = ".".join(des.split(".")[:-1])
        fileext = des.split(".")[-1]
    except IndexError as e:
        raise IndexError("No . in filename. Error: " + str(e))
    for n in range(0, split_count):
        split_args = []
        if n == 0:
            split_start = 0
        else:
            split_start = split_length * n
        split_args += ["-ss", str(split_start), "-t", str(split_length),
                       filebase + "-" + str(n + 1) + "-of-" +
                       str(split_count) + "." + fileext]
        subprocess.check_output(split_cmd + split_args)

# Loop through the filesystem
for root, dirs, files in os.walk(source_file_dir, topdown=False):
    # Loop through files
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
          split_by_seconds(source_file_dir+name, clip_duration, des_file_dir+name)
          
          # clip = VideoFileClip(source_file_dir+name)
          # video_duration = clip.duration
          # for i in range(0, int(video_duration-video_duration%clip_duration), clip_duration):
          #   clip_file_name = name[:-4]+f"{int(i/clip_duration)+1:02}"+".mp4"
          #   ffmpeg_extract_subclip(source_file_dir+name, i, i+clip_duration, targetname=des_file_dir+clip_file_name)

