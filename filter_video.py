import keras
import os
import numpy as np
import librosa
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import subprocess, math, shlex
import cv2
import time

# starting time
start = time.time()

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

actual_classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

model1 = keras.models.load_model('/media/asif/Programming/Thesis/Video_Classifier/Models/video_classification_with_audio_1.h5')
model2 = keras.models.load_model('/media/asif/Programming/Thesis/Video_Classifier/Models/video_classification_with_video_1.h5')

from tensorflow import keras

clip_duration = 5
video_classes = ['Educational', 'Gaming', 'News', 'Scenery', 'Songs', 'Sports']

source_file_dir = '/media/asif/Programming/Thesis/Video_Classifier/Filter_Video/Video_with_add/'
temp_file_dir = '/media/asif/Programming/Thesis/Video_Classifier/Filter_Video/Temp/'
des_file_dir = '/media/asif/Programming/Thesis/Video_Classifier/Filter_Video/Video_without_add/'

# def ceildiv(a, b):
#     return int(math.ceil(a / float(b)))

# def get_video_length(filename):
#     output = subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
#                                       "default=noprint_wrappers=1:nokey=1", filename)).strip()
#     video_length = int(float(output))
#     return video_length

# def split_by_seconds(filename, split_length, des, vcodec="copy", acodec="copy",
#                      extra="", video_length=None, **kwargs):
#     if split_length and split_length <= 0:
#         raise SystemExit
#     if not video_length:
#         video_length = get_video_length(filename)
#     split_count = ceildiv(video_length, split_length)
#     if split_count == 1:
#         raise SystemExit
#     split_cmd = ["ffmpeg", "-i", filename, "-vcodec", vcodec, "-acodec", acodec] + shlex.split(extra)
#     try:
#         filebase = ".".join(des.split(".")[:-1])
#         fileext = des.split(".")[-1]
#     except IndexError as e:
#         raise IndexError("No . in filename. Error: " + str(e))
#     for n in range(0, split_count):
#         split_args = []
#         if n == 0:
#             split_start = 0
#         else:
#             split_start = split_length * n
#         split_args += ["-ss", str(split_start), "-t", str(split_length),
#                        filebase[:-2] + "_" + str(n + 1) + "." + fileext]
#         subprocess.check_output(split_cmd + split_args)

# Loop through the filesystem
for root, dirs, files in os.walk(source_file_dir, topdown=False):
    # Loop through files
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
            # split_by_seconds(source_file_dir+name, clip_duration, des_file_dir+name)
            clip = VideoFileClip(source_file_dir+name)
            video_duration = clip.duration
            for i in range(0, int(video_duration-video_duration%clip_duration), clip_duration):
                clip_file_name = name[:-5]+f"{int(i/clip_duration)+1:02}"+".mp4"
                ffmpeg_extract_subclip(source_file_dir+name, i, i+clip_duration, targetname=temp_file_dir+clip_file_name)

# Loop through the filesystem
for root, dirs, files in os.walk(temp_file_dir, topdown=False):
    # Loop through files
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
            # Using ffmpeg to convert the mp4 in wav
            # Example command: "ffmpeg -i C:/test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
            command = "ffmpeg -i " + root + "/" + name + " " + "-ab 160k -ac 2 -ar 44100 -vn " + temp_file_dir + name[:-3] + "wav"
            # Execute conversion
            try:
                subprocess.call(command, shell=True)
            # Skip the file in case of error
            except ValueError:
                continue

# Loop through the filesystem
x_cnn = []
for subdir, dirs, files in os.walk(temp_file_dir, topdown=False):
    # Loop through files
    files.sort()
    for name in files:
        if name.endswith('.wav'):
            #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
            X, sample_rate = librosa.load(os.path.join(subdir, name), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
            x_cnn.append(np.expand_dims(mfccs, axis=1))
x_cnn = np.asarray(x_cnn)
predict_x = model1.predict(x_cnn) 
classes_x1 = np.argmax(predict_x,axis=1)


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x - min_dim)//2
    start_y = (y - min_dim)//2
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")
feature_extractor = build_feature_extractor()

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
    return frame_features, frame_mask    

# Loop through the filesystem
classes_x2 = []
for subdir, dirs, files in os.walk(temp_file_dir, topdown=False):
    # Loop through files
    files.sort()
    for name in files:
        if name.endswith('.mp4'):
            frames = load_video(temp_file_dir+name)
            frame_features, frame_mask = prepare_single_video(frames)
            probabilities = model2.predict(frame_features)[0]
            classes_x2.append(np.argmax(probabilities))
classes_x2 = np.asarray(classes_x2)
video_class = np.bincount(np.concatenate([classes_x1, classes_x2])).argmax()
print(actual_classes)
print(classes_x1)
print(classes_x2)
print(video_class)

clips = []
for subdir, dirs, files in os.walk(temp_file_dir, topdown=False):
    # Loop through files
    files.sort()
    i = 0
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
            if classes_x1[i]==video_class or classes_x2[i]==video_class:
                print(i, actual_classes[i])
                clips.append(VideoFileClip(temp_file_dir+name))
            i += 1
result_clip = concatenate_videoclips(clips)
result_clip.write_videofile(des_file_dir + 'video.mp4')

for subdir, dirs, files in os.walk(temp_file_dir):
    for name in files:
        os.remove(temp_file_dir+name)

# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")
