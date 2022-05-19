from traceback import print_tb
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import numpy as np
import cv2

source_file_dir = '/media/asif/Programming/Thesis/Video_Classifier/Filter_Video/Input/'
des_file_dir = '/media/asif/Programming/Thesis/Video_Classifier/Filter_Video/Output/'

clips = []
# Loop through the filesystem
for root, dirs, files in os.walk(source_file_dir, topdown=False):
    # Loop through files
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
            clips.append(VideoFileClip(source_file_dir+name))
result_clip = concatenate_videoclips(clips)
result_clip.write_videofile(des_file_dir + 'merged_video.mp4')

