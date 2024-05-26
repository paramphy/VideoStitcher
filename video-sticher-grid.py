import cv2
import numpy as np
import tqdm
import os
from moviepy.editor import ImageSequenceClip

class VideoStitcher:
    def __init__(self, video_in_paths, video_out_path, grid_shape=(4, 2), video_out_width=1280, display=True ):
        self.video_in_paths = video_in_paths
        self.video_out_path = video_out_path
        self.grid_shape = grid_shape
        self.video_out_width = video_out_width
        self.display = display

    def stitch(self, frames):
        grid_rows, grid_cols = self.grid_shape
        frame_height, frame_width = frames[0].shape[:2]

        # Create an empty canvas for the grid
        grid_height = frame_height * grid_rows
        grid_width = frame_width * grid_cols
        grid = np.zeros((grid_height, grid_width, 3), dtype="uint8")

        for idx, frame in enumerate(frames):
            row = idx // grid_cols
            col = idx % grid_cols
            grid[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width] = frame

        return grid

    def run(self):
        video_captures = [cv2.VideoCapture(path) for path in self.video_in_paths]
        n_frames = min(int(vc.get(cv2.CAP_PROP_FRAME_COUNT)) for vc in video_captures)
        fps = int(video_captures[0].get(cv2.CAP_PROP_FPS))
        frames = []

        for _ in tqdm.tqdm(range(n_frames)):
            frame_set = []
            for vc in video_captures:
                ret, frame = vc.read()
                if ret:
                    frame_set.append(frame)
                else:
                    frame_set.append(np.zeros_like(frame_set[0]))

            stitched_frame = self.stitch(frame_set)
            stitched_frame = cv2.resize(stitched_frame, (self.video_out_width, int(stitched_frame.shape[0] * (self.video_out_width / stitched_frame.shape[1]))))
            frames.append(stitched_frame)

            if self.display:
                cv2.imshow("Result", stitched_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

        # Save video
        print('[INFO]: Saving {} in {}'.format(self.video_out_path.split('/')[-1], os.path.dirname(self.video_out_path)))
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(self.video_out_path, codec='mpeg4', audio=False, verbose=False)
        print('[INFO]: {} saved'.format(self.video_out_path.split('/')[-1]))

# Example call to 'VideoStitcher'
video_paths = [
    'upper-left.mp4', 'upper-right.mp4',
    'lower-left.mp4', 'lower-right.mp4'
]
stitcher = VideoStitcher(video_in_paths=video_paths, video_out_path='grid_output.mp4')
stitcher.run()
