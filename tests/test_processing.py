from unittest.mock import patch

import pytest

from utils.processing import (
    download_video,
    extract_audio_from_video,
    link_target_ss,
    remove_unwanted_headers,
    timestamp_to_rounded_seconds,
)


@pytest.fixture
def mock_logger(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_ffmpeg_input(mocker):
    mock_ffmpeg_input = mocker.patch("ffmpeg.input")
    mocker.patch("ffmpeg.output")
    mocker.patch("ffmpeg.run")
    return mock_ffmpeg_input


@pytest.fixture
def mock_youtube_dl(mocker):
    with patch("yt_dlp.YoutubeDL") as mock_youtube_dl:
        mock_youtube_dl_inst = mocker.MagicMock(
            format="bestvideo[ext=mp4][vcodec^=avc][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
            outtmpl="video.mp4",
        )
        mock_youtube_dl.return_value = mock_youtube_dl_inst
        yield mock_youtube_dl_inst


def test_download_video_success(mock_youtube_dl):
    # arrange
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    output_dir = "videos"
    format = "bestvideo[ext=mp4][vcodec^=avc][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best"

    # act
    download_video(url, output_dir)

    # assert
    assert mock_youtube_dl.format == format
    assert mock_youtube_dl.outtmpl == "video.mp4"
    mock_youtube_dl.download.assert_called_once_with([url])


def test_extract_audio_from_video(mocker, mock_ffmpeg_input):
    # arrange
    input_dir = "test"
    output_dir = "test"
    mock_input_node = mocker.MagicMock()
    mock_output_node = mocker.MagicMock()
    mock_ffmpeg_input.return_value = mock_input_node
    mock_input_node.output.return_value = mock_output_node

    # act
    extract_audio_from_video(input_dir, output_dir)

    # assert
    mock_ffmpeg_input.assert_called_with(f"{input_dir}/video.mp4")
    mock_input_node.output.assert_any_call(
        f"{output_dir}/audio.wav", format="wav", acodec="pcm_s16le", ar=16000, ac=2
    )
    mock_output_node.run.assert_any_call(overwrite_output=True)


def test_timestamp_to_rounded_seconds():
    # arrange
    first_timestamp = "10:00:00,000"
    second_timestamp = "00:00:10,000"
    expected_first_result = 3600
    expected_second_result = 1

    # act
    first_result = timestamp_to_rounded_seconds(first_timestamp)
    second_result = timestamp_to_rounded_seconds(second_timestamp)

    # assert
    assert first_result == expected_first_result
    assert second_result == expected_second_result


def test_timestamp_to_rounded_seconds_error():
    # arrange
    invalid_timestamp = "000:00:0,00"

    # act
    with pytest.raises(ValueError, match="Invalid timestamp format."):
        timestamp_to_rounded_seconds(invalid_timestamp)


def test_link_target_ss():
    # arrange
    content = '<img src="00_00_10.jpg">\n<img src="00_30_10.jpg" alt="Testing!">'
    expected_content = '<img src="./frames/unlabelled/frame_0002.png">\n<img src="./frames/unlabelled/frame_0182.png" alt="Testing!">'

    # act
    modified_content = link_target_ss(content)

    # assert
    assert modified_content == expected_content


def test_remove_unwanted_headers():
    # arrange
    content = "## Conclusion\nThis is a test conclusion.\n## Summary\nWe need to better summarise this."
    expected_content = (
        "\nThis is a test conclusion.\n\nWe need to better summarise this."
    )

    # act
    content = remove_unwanted_headers(content)

    # assert
    assert content == expected_content
