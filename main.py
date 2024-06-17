import json
import os
from functools import reduce
from typing import Optional

import srt
from openai import OpenAI
from tqdm import tqdm

from utils.cli import cli_parser
from utils.completions import (
    generate_md_section,
    generate_transcript_segments,
    proofread_and_finalize_content,
)
from utils.logger import setup_logging
from utils.processing import (
    convert_images_to_base64,
    download_video,
    extract_audio_from_video,
    extract_ss_frames,
    generate_transcript,
    link_target_ss,
    load_images,
    load_segments,
    load_transcript,
    remove_unwanted_headers,
    retrieve_frames_by_segments,
    retrieve_transcript_by_segments,
)

logger = setup_logging(verbose=False)

DEFAULT_WORKING_DIR = "."
DEFAULT_SAVED_FRAMES_DIR = "frames"
DEFAULT_SAVED_MD_DIR = "temp"
DEFAULT_SAVED_TRANSCRIPT_PATH = "transcript.srt"


def main(
    url: str,
    transcript_path: str,
    segments_path: Optional[str],
    api_key: str,
    semantic_chunking: bool,
):
    client = OpenAI(api_key=api_key)

    logger.info("Running processing to download and extract pre-requisities")
    # pre-requisites, skip if resuming workflow (assume resuming means pre-requisites available)
    download_video(url=url, output_dir=DEFAULT_WORKING_DIR)
    extract_audio_from_video(
        input_dir=DEFAULT_WORKING_DIR, output_dir=DEFAULT_WORKING_DIR
    )
    extract_ss_frames(
        input_dir=DEFAULT_WORKING_DIR, output_dir=DEFAULT_SAVED_FRAMES_DIR
    )

    if transcript_path:
        transcript = load_transcript(transcript_path)
    else:
        generate_transcript(
            input_dir=DEFAULT_WORKING_DIR, output_dir=DEFAULT_WORKING_DIR
        )
        transcript = load_transcript(DEFAULT_SAVED_TRANSCRIPT_PATH)

    if segments_path:
        segments = load_segments(segments_path)
    else:
        segments = generate_transcript_segments(client, transcript)
        # save segments for re-runs
        with open("segments.json", "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=4)
        segments = json.loads(segments)

    logger.info(f"{len(segments)} segments identified")

    # match transcript content by segment
    parsed_transcript = list(srt.parse(transcript))
    segments = retrieve_transcript_by_segments(segments, parsed_transcript)
    # automatically created from steps above
    labelled_imgs = load_images(f"{DEFAULT_SAVED_FRAMES_DIR}/labelled")
    segments = retrieve_frames_by_segments(
        segments=segments,
        images=labelled_imgs,
        semantic_chunking=semantic_chunking,
    )

    section_counter = 0
    section_contents = []
    logger.info("Generating markdown content on segment-level")
    for seg in tqdm(segments):
        seg_transcript = seg["transcript"]
        # no generation if transcript content
        if not (seg_transcript.strip()):
            continue
        b64_imgs = convert_images_to_base64(seg["images"])

        content = generate_md_section(client, seg_transcript, b64_imgs)
        processing_funcs = [link_target_ss, remove_unwanted_headers]
        content = reduce(lambda x, fn: fn(x), processing_funcs, content)
        content = proofread_and_finalize_content(client, content)
        section_contents.append(content)

        # store md content for each section for vetting and retry purpose
        if not os.path.exists(DEFAULT_SAVED_MD_DIR):
            os.makedirs(DEFAULT_SAVED_MD_DIR)
            logger.info(f"Directory created: {DEFAULT_SAVED_MD_DIR}")
        with open(f"temp/section_{section_counter}.md", "w") as f:
            f.write(content)
        section_counter += 1

    logger.info("Generating final markdown")
    final_content = "\n".join(section_contents)
    with open("index.md", "w", encoding="utf-8") as f:
        f.write(final_content)


if __name__ == "__main__":
    args = cli_parser()
    main(
        transcript_path=args.transcript_path,
        segments_path=args.segments_path,
        api_key=args.api_key,
        url=args.url,
        semantic_chunking=args.semantic_chunking,
    )
