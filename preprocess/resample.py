import sys

from scipy.ndimage.interpolation import zoom
import imageio

def downsample_video(video, output_frames):
    output_vid = zoom(video, [(float(output_frames)/video.shape[0]),1,1,1])
    return output_vid

def read_video_to_frames(filename, output_resolution = (150,150)):
    vid = imageio.get_reader(filename,  'ffmpeg')
    vid2 = imageio.get_reader(filename,  'ffmpeg')
    frames = []
    no_of_frames = vid.get_length()
    input_resolution = np.array(vid2.get_data(0)).shape
    for i in range(0,no_of_frames,1):
        frames.append(zoom(np.array(vid.get_data(i)), [(output_resolution[0]/float(input_resolution[0])),(output_resolution[1]/float(input_resolution[1])),1]))
    frames = np.array(frames)
    print frames.shape
    return frames


if __name__ == '__main__':
    read_video_to_frames(sys.argv[1])
