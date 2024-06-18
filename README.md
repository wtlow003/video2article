<h1 align="center">Video2Article</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.9.10-orange"
         alt="python version">
     <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"
          alt="uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json"
         alt="ruff">
    <!-- <a href="https://github.com/wtlow003/auto-dubs/actions/workflows/python-app.yml">
     <img src="https://github.com/wtlow003/auto-dubs/actions/workflows/python-app.yml/badge.svg" alt="pytest">
    </a> -->
    <!-- <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"
         alt="docker"> -->
</p>

<p align="center">
    <a href=#about>About</a> •
    <a href=#getting-started>Getting Started</a> •
    <a href=#usage>Usage</a>
</p>

## About

![video2article-output](/assets/video-2-article-small.gif)

**Video2Article** demonstrates the use of Large Multimodal Model (LMM) to generate a full-length article from a video tutorial.

Using the vision capabilities of `GPT-4o`, you can now turn any video tutorial into technical article with relevant code snippets, screenshots extracted from the video without manual intervention.

For specifics in the implementation, you can read more in my detailed [write-up](https://wtlow003.github.io/posts/transforming-video-to-article-with-gpt-4o/).

> [!NOTE]
>
> While Video2Article works well to a certain extent, it still requires manual proofreading and editing to fix inaccuracies and inconsistencies in the content and formatting.

## Getting Started

### Setting Up Environment

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. To install `uv`, please refer to this [guide](https://github.com/astral-sh/uv#getting-started):

```shell
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
pip install uv

# With pipx.
pipx install uv

# With Homebrew.
brew install uv

# With Pacman.
pacman -S uv
```

To setup the project and install the required dependencies:

```shell
# git clone the repo along with submodules
git clone --recurse-submodules https://github.com/wtlow003/video2article.git

# create a virtual env
uv venv

# install dependencies
uv pip install -r requirements.txt  # Install from a requirements.txt file.
```

## Usage

The following are the available options to trigger a dubbing workflow:

```shell
source .venv/bin/activate
python3 main.py --help

>>> usage: main.py [-h] [--api-key API_KEY] [--transcript-path TRANSCRIPT_PATH] [--segments-path SEGMENTS_PATH] [--url URL]
               [--semantic-chunking]

Convert video to article.

optional arguments:
  -h, --help            show this help message and exit
  --api-key API_KEY     OpenAI API key.
  --transcript-path TRANSCRIPT_PATH
                        [OPTIONAL] Path to video transcript (in SRT) format.
  --segments-path SEGMENTS_PATH
                        [OPTIONAL] Path to transcript segments (in JSON) format.
  --url URL             Video url.
  --semantic-chunking   Enable semantic chunking of images with Semantic Router.
```

For example, to trigger a straightforward article generation from a YouTube url:

```shell
# api keys for openai + langsmith tracing
source .env

python3 main.py --url "https://www.youtube.com/watch?v=TCH_1BHY58I" --semantic-chunking 
```