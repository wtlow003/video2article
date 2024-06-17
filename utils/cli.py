import argparse


def cli_parser():
    """CLI Arguments Parser for Video2Article"""
    parser = argparse.ArgumentParser(description="Convert video to article.")
    parser.add_argument("--api-key", type=str, help="OpenAI API key.")
    parser.add_argument(
        "--transcript-path",
        type=str,
        help="[OPTIONAL] Path to video transcript (in SRT) format.",
    )
    parser.add_argument(
        "--segments-path",
        type=str,
        help="[OPTIONAL] Path to transcript segments (in JSON) format.",
    )
    parser.add_argument("--url", type=str, help="Video url.")
    parser.add_argument(
        "--semantic-chunking",
        action="store_true",
        help="Enable semantic chunking of images with Semantic Router.",
    )
    args = parser.parse_args()
    return args
