import argparse
import pickle
import os.path as osp
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def load_shots(shot_path: str) -> List[Tuple[int, int]]:
    """Load shot boundaries txt file."""
    with open(shot_path, "r") as f:
        raw_shots = f.read()
    split_shots = raw_shots.split("\n")

    parsed_shots = []
    for str_shot in split_shots:
        if len(str_shot):
            str_start, str_end = str_shot.split(" ")
            parsed_shots.append([int(str_start), int(str_end)])

    return parsed_shots


def load_tracks(track_path: str) -> Dict[str, Any]:
    """Load character tracks pickle file."""
    with open(track_path, "rb") as f:
        raw_tracks = pickle.load(f)

    return raw_tracks


def draw_track(
    frame_index: int, frame: np.array, track: Dict[str, Any]
) -> np.array:
    """Draw the corresponding bounding-box on the frame."""
    (box_index,) = np.where(track[:, 0] == frame_index)
    bbox = track[box_index].reshape(-1).astype(int)
    if bbox.size > 0:
        frame = cv2.rectangle(
            frame,
            (bbox[1], bbox[2]),
            (bbox[3], bbox[4]),
            (0, 0, 255),
            thickness=5,
        )
    return frame


def load_frame_bb(
    frame_dir: str, shot_start: int, shot_end: int, track: Dict[str, Any]
) -> List[np.array]:
    """Load shot frames and draw bounding-boxes."""
    annotated_frames = []
    for frame_index in range(shot_start, shot_end + 1):
        frame_path = osp.join(frame_dir, str(frame_index).zfill(6) + ".jpg")
        frame = cv2.imread(frame_path)
        annotated_frames.append(draw_track(frame_index, frame, track))

    return annotated_frames


def track_shot_character(
    frame_dir: str,
    episode_name: str,
    shots: List[Tuple[int, int]],
    shot_index: int,
    tracks: Dict[str, Any],
    character_name: str,
):
    """
    Given an episode shot index, display tracks of the specified character.
    """
    frame_episode_dir = osp.join(frame_dir, episode_name)
    shot_start, shot_end = shots[shot_index]
    for track_index, track in tracks[episode_name]["body"].items():
        frame_start, frame_end = track[0][0], track[-1][0]
        if (frame_start >= shot_start) and (frame_end <= shot_end):
            if tracks[episode_name]["GT"][track_index] == character_name:
                annotated_frames = load_frame_bb(
                    frame_episode_dir, shot_start, shot_end, track
                )
                break

    return annotated_frames


def write_clip(frames: List[np.array], output_filename: str, fps: float):
    """Write a clip in `mp4` format from a list of frames.

    :param frames: 1st dim frame index, 2nd dim frame to be stacked
        together (must be the same dimensions).
    :param output_filename: file name of the saved output.
    :param fps: wanted frame per second rate.
    """
    frame_height, frame_width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Initialize the video writer
    clip = cv2.VideoWriter(
        output_filename,
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    # Write each frame
    for frame in frames:
        clip.write(frame)

    # Release the video writer
    clip.release()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "shot_path", type=str, help="Path to the txt shot file"
    )
    parser.add_argument(
        "track_path", type=str, help="Path to the pickle track file"
    )
    parser.add_argument(
        "frame_dir", type=str, help="Path to the directory containing frames"
    )
    parser.add_argument(
        "episode_name", type=str, help="Name of the episode to process"
    )
    parser.add_argument(
        "shot_index", type=int, help="Index of the shot to process"
    )
    parser.add_argument(
        "character_name", type=str, help="Name of the character to track"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    shot_path = args.shot_path
    track_path = args.track_path
    frame_dir = args.frame_dir
    episode_name = args.episode_name
    shot_index = args.shot_index
    character_name = args.character_name

    shots = load_shots(shot_path)
    tracks = load_tracks(track_path)

    annotated_frames = track_shot_character(
        frame_dir,
        episode_name,
        shots,
        shot_index,
        tracks,
        character_name,
    )

    # write_clip(annotated_frames, "output.mp4", 25)
