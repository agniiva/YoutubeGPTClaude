import streamlit as st
import os
import shutil
import librosa
import openai
import soundfile as sf
import yt_dlp
from yt_dlp.utils import DownloadError
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize OpenAI and Claude API
openai.api_key = openai_api_key
anthropic = Anthropic(api_key=anthropic_api_key)


# Common Functions
def find_audio_files(path, extension=".mp3"):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))
    return audio_files

def youtube_to_mp3(youtube_url: str, output_dir: str) -> str:
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with yt_dlp.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        pass  # Handle error as needed

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename

def chunk_audio(filename, segment_length: int, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    audio, sr = librosa.load(filename, sr=44100)
    duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(duration / segment_length) + 1
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)
    chunked_audio_files = find_audio_files(output_dir)
    return sorted(chunked_audio_files)

def transcribe_audio(audio_files: list, output_file=None, model="whisper-1") -> list:
    transcripts = []
    for audio_file in audio_files:
        audio = open(audio_file, "rb")
        response = openai.Audio.transcribe(model, audio)
        transcripts.append(response["text"])
    if output_file is not None:
        with open(output_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript + "\n")
    return transcripts

# OpenAI Summary
def summarize_openai(chunks: list[str], system_prompt: str, model="gpt-3.5-turbo", output_file=None):
    summaries = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )
        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)
    if output_file is not None:
        with open(output_file, "w") as file:
            for summary in summaries:
                file.write(summary + "\n")
    return summaries

# Claude Summary
def summarize_claude(chunks: list[str], system_prompt: str, model="claude-2", output_file=None):
    summaries = []
    for chunk in chunks:
        prompt = f"{HUMAN_PROMPT} {system_prompt} {chunk}{AI_PROMPT}"
        completion = anthropic.completions.create(model=model, max_tokens_to_sample=300, prompt=prompt)
        summary = completion.completion
        summaries.append(summary)
    if output_file is not None:
        with open(output_file, "w") as file:
            for summary in summaries:
                file.write(summary + "\n")
    return summaries

# Main function
def summarize_youtube_video(youtube_url, outputs_dir, progress_bar, progress_text, summarization_function):
    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/chunks/"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    segment_length = 10 * 60  # 10 minutes

    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
        os.mkdir(outputs_dir)

    progress_text.text("Downloading video...")
    audio_filename = youtube_to_mp3(youtube_url, output_dir=raw_audio_dir)
    progress_bar.progress(0.25)

    progress_text.text("Chunking audio...")
    chunked_audio_files = chunk_audio(audio_filename, segment_length=segment_length, output_dir=chunks_dir)
    progress_bar.progress(0.5)

    progress_text.text("Transcribing audio...")
    transcriptions = transcribe_audio(chunked_audio_files, transcripts_file)
    progress_bar.progress(0.75)

    progress_text.text("Generating summary...")
    system_prompt = "You are a helpful assistant that summarizes and distills YouTube videos. You are provided chunks of raw audio that were transcribed from the video's audio. Summarize and distill the current chunk to succinct and clear bullet points of its contents."
    summaries = summarization_function(transcriptions, system_prompt=system_prompt, output_file=summary_file)

    system_prompt_tldr = "You are a helpful assistant that summarizes YouTube videos. Someone has already summarized the video to key points. Summarize the key points to one or two sentences that capture the essence of the video."
    long_summary = "\n".join(summaries)
    short_summary = summarization_function([long_summary], system_prompt=system_prompt_tldr, output_file=summary_file)[0]

    progress_bar.progress(1.0)
    progress_text.text("Summary complete.")

    return long_summary, short_summary

# Streamlit UI
def main():
    st.title("YouGPTube Summarizer 🦾")

    summarization_choice = st.sidebar.selectbox("Choose summarization method:", ["OpenAI LLM", "Claude"])
    if summarization_choice == "OpenAI LLM":
        summarization_function = summarize_openai
    else:
        summarization_function = summarize_claude

    youtube_url = st.text_input("Enter YouTube URL:", "")

    if st.button("Summarize Video"):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        with st.spinner("Summarizing... This might take a while."):
            outputs_dir = "outputs/"
            long_summary, short_summary = summarize_youtube_video(youtube_url, outputs_dir, progress_bar, progress_text, summarization_function)

        st.subheader("Long Summary:")
        st.write(long_summary)

        st.subheader("Video - TL;DR")
        st.write(short_summary)

if __name__ == "__main__":
    main()
