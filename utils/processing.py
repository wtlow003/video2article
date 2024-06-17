import base64
import io
import json
import os
import re
import subprocess
from datetime import timedelta
from typing import Dict, List, Union

import cv2
import ffmpeg
import numpy as np
import torch
import yt_dlp
from cv2.typing import MatLike
from PIL import Image
from semantic_chunkers import ConsecutiveChunker
from semantic_chunkers.schema import Chunk
from semantic_router.encoders import VitEncoder
from srt import Subtitle

from .logger import setup_logging

UNWANTED_HEADERS = [
    "## Conclusion",
    "### Conclusion",
    "#### Conclusion",
    "## Summary",
    "### Summary",
    "#### Summary",
]

logger = setup_logging(verbose=False)


def extract_audio_from_video(input_dir: str, output_dir: str):
    """Extract .wav audio from downloaded video.

    Args:
        input_dir (str): Input directory for video.
        output_dir (str): Output directory for audio.
    """

    logger.info("Extracting audio from input video")

    output_audio_path = f"{output_dir}/audio.wav"

    (
        ffmpeg.input(f"{input_dir}/video.mp4")
        .output(output_audio_path, format="wav", acodec="pcm_s16le", ar=16000, ac=2)
        .run(overwrite_output=True)
    )


def download_video(url: str, output_dir: str):
    """Download video given URL.

    Args:
        url (str): Video URL.
        output_dir (str): Output directory for video.
    """

    ydl_opts = {
        "format": "bestvideo[ext=mp4][vcodec^=avc][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
        "outtmpl": f"{output_dir}/video.%(ext)s",
        "logger": logger,
    }

    ydl = yt_dlp.YoutubeDL(ydl_opts)
    try:
        logger.info(f"Downloading video from: {url}")
        ydl.download([url])
    except Exception as e:
        logger.error(f"Failed to download video from: {url}")
        raise (e)

    logger.info(f"Video downloaded to: {output_dir}")


def generate_transcript(input_dir: str, output_dir: str):
    """Generate transcript (in SRT format) with whisper.cpp.

    Args:
        input_dir (str): Input directory for audio.
        output_dir (str): Outpout directory for transcript.
    """

    logger.info("Generating transcript")

    build_cmd = [
        "bash",
        "./models/download-ggml-model.sh",
        "small.en",
    ]
    generate_cmd = [
        "./whisper.cpp/main",
        "-m",
        "./whisper.cpp/models/ggml-small.en.bin",
        "-f",
        f"{input_dir}/audio.wav",
        "-osrt",
        "-of",
        f"{output_dir}/transcript",
        "-p",
        "8",
        "-l",
        "auto",
    ]
    try:
        _ = subprocess.run(
            build_cmd,
            cwd="./whisper.cpp",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        _ = subprocess.run(
            generate_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating transcript: {e.stderr}")
        raise (e)


def extract_ss_frames(input_dir: str, output_dir: str):
    """Extract screenshot frames from video with ffmpeg.

    Args:
        input_dir (str): Input directory of video.
        output_dir (str): Output directory for video frames.
    """

    directories = [f"{output_dir}/labelled", f"{output_dir}/unlabelled"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directory created: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

    labelled_cmd = [
        "ffmpeg",
        "-i",
        f"{input_dir}/video.mp4",
        "-vf",
        r"fps=1/10,drawtext=text='%{pts\:gmtime\:0\:%H\\\:%M\\\:%S}':x=10:y=10:fontcolor=white:fontsize=96:box=1:boxcolor=black@1.0",
        "-vsync",
        "vfr",
        f"{output_dir}/labelled/frame_%04d.png",
    ]
    unlabelled_cmd = [
        "ffmpeg",
        "-i",
        f"{input_dir}/video.mp4",
        "-vf",
        "fps=1/10",
        "-vsync",
        "vfr",
        f"{output_dir}/unlabelled/frame_%04d.png",
    ]
    try:
        _ = subprocess.run(
            labelled_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        _ = subprocess.run(
            unlabelled_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting frames: {e.stderr}")
        raise (e)


def load_transcript(path: str) -> str:
    """Load transcript.

    Args:
        path (str): Path to transcript.

    Returns:
        str: Parsed transcript string.
    """

    with open(path, "r") as f:
        transcript = f.read()
    return transcript


def load_segments(path: str) -> List[Dict[str, str]]:
    """Load chunked transcript segments.

    Args:
        path (str): Path to transcript segments.

    Returns:
        List[Dict[str, str]]: List of transcript segments.
    """

    logger.info(f"Loading segments from {path}")

    with open(path, "r") as f:
        segments = json.loads(json.load(f))
    return segments


def load_images(path: str) -> List[MatLike]:
    """Load video frames.

    Args:
        path (str): Path to video frames.
    """

    logger.info("Loading frames")

    images = []
    files = os.listdir(path)
    for filename in sorted(files):
        im_path = os.path.join(path, filename)
        if im_path.lower().endswith((".png", ".jpg", ".jpeg")):
            im = cv2.imread(im_path)
            if im is not None:
                images.append(im)
    logger.info(f"Loaded {len(images)} frames")
    return images


def parse_time_to_timedelta(time_str: str) -> timedelta:
    """Parse timestamp string to timedelta.

    Args:
        time_str (str): Timestamp in HH:MM:SS,MS format.

    Raises:
        ValueError: If timestamp is not in HH:MM:SS,MS format.

    Returns:
        timedelta: Timestamp in timedelta format.
    """

    # hh:mm:ss,ms
    pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"

    # match the pattern
    match = re.match(pattern, time_str)
    if not match:
        raise ValueError("Invalid time format. Ensure it matches 'HH:MM:SS,SSS'")

    hours, minutes, seconds, milliseconds = match.groups()
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)

    milliseconds_in_seconds = milliseconds / 1000.0

    # Create and return the timedelta object
    time_delta = timedelta(
        hours=hours, minutes=minutes, seconds=seconds + milliseconds_in_seconds
    )
    return time_delta


def retrieve_transcript_by_segments(
    segments: List[Dict[str, str]], transcript: List[Subtitle]
) -> List[Dict[str, str]]:
    """Match transcript content within segment time boundary.

    Args:
        segments (List[Dict[str, str]]): List of segments.
        transcript (List[Subtitle]): Parsed transcript in SRT format.

    Returns:
        List[Dict[str, str]]: List of segments with transcript information.
    """

    logger.info("Retrieving transcript on segment-level")

    for seg in segments:
        seg_transcripts = []
        start_td = parse_time_to_timedelta(seg["start"])
        end_td = parse_time_to_timedelta(seg["end"])

        for trans in transcript:
            if (trans.start >= start_td) and (trans.end <= end_td):
                seg_transcripts.append(trans.content.strip())
        seg["transcript"] = " ".join(seg_transcripts)
    return segments


def semantic_chunker(image_frames: List[MatLike]) -> List[Chunk]:
    """Chunker to chunk frames semantically with ViT.

    Args:
        image_frames (List[MatLike]): List of frames.

    Returns:
        List[Chunk]: List of frames in chunks.
    """

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using {device} for semantic chunking")
    encoder = VitEncoder(device=device)
    chunker = ConsecutiveChunker(encoder=encoder, score_threshold=0.95)
    frames = list(map(Image.fromarray, image_frames))
    chunks = chunker(docs=[frames])

    return chunks[0]


def retrieve_frames_by_segments(
    segments: List[Dict], images: List[MatLike], semantic_chunking: bool
) -> List[Dict]:
    """Match frames to segment.

    There are two approach to matching frames:
        1. The `semantic` way: Using semantic-chunker to match frames
        based on features extracted with ViT
        2. The `naive` way: Select frames in interval to ensure only max N
        number of frames is selected.

    Args:
        segments (List[Dict]): List of segments.
        images (List[MatLike]): List of timestamped frames.
        semantic_chunking (bool): Whether to use semantinc-chunker for frames selection.

    Returns:
        List[Dict]: List of segments with selected frames.
    """

    logger.info("Retrieving frames on segment-level")

    for idx, seg in enumerate(segments):
        seg_imgs = []
        start_second = timestamp_to_rounded_seconds(seg["start"])
        end_second = timestamp_to_rounded_seconds(seg["end"])
        for img_idx in range(start_second, min(end_second, len(images))):
            seg_imgs.append(images[img_idx])
        if semantic_chunking:
            semantic_chunked_seg_imgs = []
            # semantic chunking of image via ViTEncoder
            chunks = semantic_chunker(seg_imgs)
            for chunk in chunks:
                semantic_chunked_seg_imgs.append(chunk.splits[0])
            seg_imgs = semantic_chunked_seg_imgs
        # only retrieve ~10-15 images per segment
        # act as another round of filter
        interval = (end_second - start_second) // 10
        seg_imgs = seg_imgs[:: interval if interval else 1]
        seg["images"] = seg_imgs
        logger.info(f"{len(seg_imgs)} retrieved for segment: {idx+1}")
    return segments


def timestamp_to_rounded_seconds(timestamp: str) -> int:
    """Round timestamp to nearest 10 seconds.

    Args:
        timestamp (str): Timestamp in HH:MM:SS,MS format.

    Raises:
        ValueError: If timestamp is not in HH:MM:SS,MS format.

    Returns:
        int: Timestamp in rounded 10 seconds.
    """

    # hh:mm:ss,ms
    pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    match = re.match(pattern, timestamp)
    if not match:
        raise ValueError("Invalid timestamp format. Ensure it matches 'HH:MM:SS,SSS'")

    hours, minutes, seconds, _ = map(int, match.groups())

    # convert entire timestamp to seconds
    total_seconds = (hours * 3600) + (minutes * 60) + seconds

    rounded_seconds = round(total_seconds // 10)

    return rounded_seconds


def convert_images_to_base64(images: List[Union[MatLike, np.ndarray]]) -> List[str]:
    """Convert frames to base64 string.

    Args:
        images (List[Union[MatLike, np.ndarray]]): List of images.

    Returns:
        List[str]: List of base64-encoded frames.
    """

    b64_imgs = []
    for im in images:
        # handle semantic chunker which requires nd.array setup
        if isinstance(im, Image.Image):
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG")
            buffer = buffer.getvalue()
        else:
            _, buffer = cv2.imencode(".jpg", im)
        b64_imgs.append(base64.b64encode(buffer).decode("utf-8"))  # type: ignore
    return b64_imgs


def link_target_ss(content: str) -> str:
    """Match actual frames to placeholder image url in content.

    Args:
        content (str): Raw conten generated by LLM/LMM.

    Returns:
        str: Edited content.
    """

    pattern = r'<img\s+src=["\'](\d{2}_\d{2}_\d{2})\.jpg["\']'
    img_tags = re.findall(pattern, content, re.IGNORECASE)

    for tag in img_tags:
        hh, mm, ss = map(int, tag.split("_"))
        # +1 to as frames start from frame_0001, frame_0002
        # 00:00:00 -> frame_0001
        target = (((hh * 3600) + (mm * 60) + ss) // 10) + 1
        content = content.replace(
            f"{tag}.jpg", f"./frames/unlabelled/frame_0{target:03}.png"
        )

    return content


def remove_unwanted_headers(content: str) -> str:
    """Removing unwanted headers from content.

    Args:
        content (str): Raw content generated by LLM/LMM.

    Returns:
        str: Edited content.
    """

    for header in UNWANTED_HEADERS:
        content = content.replace(header, "")
    return content
