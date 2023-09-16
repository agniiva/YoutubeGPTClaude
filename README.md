# Youtube Summarizer with Claude-2, OpenAI & Whisper

## Overview

### What does it do?

The YouGPTube Summarizer is a Python-based application that utilizes advanced machine learning models from OpenAI and Anthropics to summarize YouTube videos. Given a YouTube URL, it downloads the video, extracts the audio, transcribes it, and then summarizes the content. The summarization can be done using either OpenAI's GPT-3.5 or Anthropics' Claude model. 

### Why is it useful?

Ever felt overwhelmed by the amount of content in a lengthy YouTube video and wished for a concise summary? The YouGPTube Summarizer can help you get the essence of a video in a fraction of the time it takes to watch it. Moreover, by using Whisper API for transcription, it can transcribe videos in multiple languages and generate summaries, thus breaking the language barrier.

### Video Demo

https://github.com/agniiva/YoutubeGPTClaude/assets/73607864/480edb2c-3fdb-4c68-9cc0-cd89b1090ed6

The Generations here are 3x fasten up! it took around 3-4 minutes for a video of 16 minutes, but its pretty accurate.


### Tech Implementation

Under the hood, the application uses several Python libraries such as `streamlit` for the web interface, `librosa` for audio processing, `openai` for transcription and summarization, and `yt_dlp` for YouTube video downloading. It has different functions to handle tasks like audio downloading, chunking, transcribing, and summarizing.

## Prerequisites

### Install ffmpeg

The program uses ffmpeg for audio processing. Make sure to install it in your system. You can install it using the package manager for your system.

For Ubuntu:

```bash
sudo apt-get install ffmpeg
```

For macOS:

```bash
brew install ffmpeg
```

### Python Dependencies

All Python dependencies are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

### API Keys

You'll need to obtain API keys for OpenAI and Anthropics (Claude). Store these keys in `.env.example` and rename the file to `.env`. 

You can also automatically rename the `.env.example` file by running the following command:

```bash
mv .env.example .env
```

## Running the Application

To run the app, navigate to the directory where the code is located and run:

```bash
streamlit run <filename>.py
```

## Code Documentation

### Importing Libraries

- `streamlit`: For creating the web interface
- `os, shutil`: For file and directory operations
- `librosa`: For audio processing
- `openai`: For OpenAI API calls
- `soundfile as sf`: For audio file processing
- `yt_dlp`: For downloading YouTube videos
- `anthropic`: For Anthropics (Claude) API
- `dotenv`: For loading environment variables

### Functions

#### `find_audio_files(path, extension=".mp3")`

Finds all audio files in the given path with the specified extension.

#### `youtube_to_mp3(youtube_url: str, output_dir: str) -> str`

Downloads the YouTube video from the given URL and saves it as an mp3 file in the specified directory.

#### `chunk_audio(filename, segment_length: int, output_dir)`

Chunks the given audio file into segments of specified length (in seconds) and saves them in the specified directory.

#### `transcribe_audio(audio_files: list, output_file=None, model="whisper-1") -> list`

Transcribes the given audio files using OpenAI's Whisper model.

#### `summarize_openai(chunks: list[str], system_prompt: str, model="gpt-3.5-turbo", output_file=None)`

Summarizes the given list of text chunks using OpenAI's GPT-3.5 model.

#### `summarize_claude(chunks: list[str], system_prompt: str, model="claude-2", output_file=None)`

Summarizes the given list of text chunks using Anthropics' Claude model.

#### `summarize_youtube_video(youtube_url, outputs_dir, progress_bar, progress_text, summarization_function)`

Main function that orchestrates the summarization process.

### Streamlit UI (`main()`)

Streamlit interface for user inputs and displaying summaries.

## Customization

You can customize the summarization by changing the `system_prompt`. This allows you to tailor the summary to your specific needs.
