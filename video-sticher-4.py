import cv2
import numpy as np
import imutils
import tqdm
import os
from moviepy.editor import ImageSequenceClip

class VideoStitcher:
    def __init__(self, video_in_paths, video_out_path, video_out_width=1280, display=True):
        # Initialize arguments
        self.video_in_paths = video_in_paths
        self.video_out_path = video_out_path
        self.video_out_width = video_out_width
        self.display = display

        # Initialize the saved homography matrices
        self.saved_homo_matrices = [None] * (len(video_in_paths) - 1)

    def stitch(self, images, ratio=0.75, reproj_thresh=4.0):
        # Initialize the result with the first image
        result = images[0]

        for i in range(1, len(images)):
            # If the saved homography matrix is None, then we need to apply keypoint matching to construct it
            if self.saved_homo_matrices[i - 1] is None:
                # Detect keypoints and extract features
                (keypoints_a, features_a) = self.detect_and_extract(result)
                (keypoints_b, features_b) = self.detect_and_extract(images[i])

                # Match features between the two images
                matched_keypoints = self.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

                # If the match is None, then there aren't enough matched keypoints to create a panorama
                if matched_keypoints is None:
                    return None

                # Save the homography matrix
                self.saved_homo_matrices[i - 1] = matched_keypoints[1]

            # Apply a perspective transform to stitch the images together using the saved homography matrix
            output_shape = ([max(result.shape[1],images[i].shape[1]), max(result.shape[0], images[i].shape[0])])
            result = cv2.warpPerspective(result, self.saved_homo_matrices[i - 1], output_shape)
            result[0:images[i].shape[0], 0:images[i].shape[1]] = images[i]

        # Return the stitched image
        return result

    @staticmethod
    def detect_and_extract(image):
        # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
        descriptor = cv2.SIFT_create()
        (keypoints, features) = descriptor.detectAndCompute(image, None)

        # Convert the keypoints from KeyPoint objects to numpy arrays
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        # Return a tuple of keypoints and features
        return (keypoints, features)

    @staticmethod
    def match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh):
        # Compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, k=2)
        matches = []

        for raw_match in raw_matches:
            # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(raw_match) == 2 and raw_match[0].distance < raw_match[1].distance * ratio:
                matches.append((raw_match[0].trainIdx, raw_match[0].queryIdx))

        # Computing a homography requires at least 4 matches
        if len(matches) > 4:
            # Construct the two sets of points
            points_a = np.float32([keypoints_a[i] for (_, i) in matches])
            points_b = np.float32([keypoints_b[i] for (i, _) in matches])

            # Compute the homography between the two sets of points
            (homography_matrix, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_thresh)

            # Return the matches, homography matrix and status of each matched point
            return (matches, homography_matrix, status)

        # No homography could be computed
        return None

    def run(self):
        # Set up video capture for all input videos
        video_captures = [cv2.VideoCapture(path) for path in self.video_in_paths]
        print('[INFO]: Videos loaded')
        print('[INFO]: Video stitching starting....')

        # Get information about the videos
        n_frames = min(int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) for video_capture in video_captures)
        fps = int(video_captures[0].get(cv2.CAP_PROP_FPS))
        frames = []

        for _ in tqdm.tqdm(np.arange(n_frames)):
            # Grab the frames from their respective video streams
            frames_to_stitch = [vc.read()[1] for vc in video_captures]

            if None not in zip(frames_to_stitch):
                # Stitch the frames together to form the panorama
                stitched_frame = self.stitch(frames_to_stitch)

                # No homography could be computed
                if stitched_frame is None:
                    print("[INFO]: Homography could not be computed!")
                    break

                # Add frame to video
                stitched_frame = imutils.resize(stitched_frame, width=self.video_out_width)
                frames.append(stitched_frame)

                if self.display:
                    # Show the output images
                    cv2.imshow("Result", stitched_frame)

                # If the 'q' key was pressed, break from the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

        # Save video
        print('[INFO]: Saving {} in {}'.format(self.video_out_path.split('/')[-1],
                                               os.path.dirname(self.video_out_path)))
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(self.video_out_path, codec='mpeg4', audio=False, verbose=False)
        print('[INFO]: {} saved'.format(self.video_out_path.split('/')[-1]))

# Example call to 'VideoStitcher'
video_paths = [
    'upper-left.mp4', 'upper-right.mp4',
    'lower-left.mp4', 'lower-right.mp4'
]
stitcher = VideoStitcher(video_in_paths=video_paths, video_out_path='stitched_output.mp4')
stitcher.run()
