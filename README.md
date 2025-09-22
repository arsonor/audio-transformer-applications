# Hugging Face Audio Course – Notebooks

This repository contains my hands-on work following the [Hugging Face Audio Course](https://huggingface.co/learn/audio-course).  
Each notebook demonstrates a key concept or application in audio machine learning.

---

## Unit 2
### [1-gentle-introduction.ipynb](./1-gentle-introduction.ipynb)
- Introduction to audio ML applications using Hugging Face pipelines.
- Demonstrated:
    - Loading a dataset in streaming mode.
    - Displaying the waveform and spectrogram of an audio example.
  - Loading pre-trained audio models.
  - Running simple inference for tasks like ASR (Automatic Speach recognition).

---

## Unit 4
### [2-audio-classifier.ipynb](./2-audio-classifier.ipynb)
- Goal: Build an audio classifier using transformer models.
- Demonstrated:
  - Running pipelines with diverse datasets for Keyword Spotting, Language Identification or Zero-Shot Audio Classification.

  - Loading and preprocessing an audio datasets for Music Classification.
  - Fine-tuning the model: https://huggingface.co/arsonor/distilhubert-finetuned-gtzan
  - and integrating it in a Gradio demo.

### [3-gtzan-music-genre-classification.ipynb](./3-gtzan-music-genre-classification.ipynb) (Hands-on exercise)
- Goal: Apply audio classification to music genres with the GTZAN dataset.
- Demonstrated:
  - Fine-tuning an AST model (Audio Spectrogram Transformer).
  - Evaluating results with accuracy and confusion matrix.
  - Achieving >87% Accuracy: https://huggingface.co/arsonor/ast-finetuned-audioset-10-10-0.4593-finetuned-gtzan

---

## Unit 5
### [4-automatic-speech-recognition.ipynb](./4-automatic-speech-recognition.ipynb)
- Goal: Perform automatic speech recognition (ASR) with pre-trained models.
- Demonstrated:
  - Using pipeline with diverse datasets for speech-to-text transcription.
  - Evaluating transcriptions with metrics like WER (Word Error Rate).
  - Fine-tuning a Whisper model on a specific language: https://huggingface.co/arsonor/whisper-small-dv

### [5-asr-model-fine-tuning.ipynb](./5-asr-model-fine-tuning.ipynb) (Hands-on exercise)
- Goal: Fine-tune the ”openai/whisper-tiny” model using the American English (“en-US”) subset of the ”PolyAI/minds14” dataset.
- Demonstrated:
  - Training a pre-trained ASR model for domain adaptation.
  - Achieving a Word Error Rate of 0.32:
  https://huggingface.co/arsonor/whisper-tiny-en

---

## Unit 6
### [6-tts-pre-trained-models.ipynb](./6-tts-pre-trained-models.ipynb)
- Goal: Explore text-to-speech (TTS) using pre-trained models.
- Demonstrated:
  - Converting text into natural-sounding speech.
  - Experimenting with different voices and vocoders: SpeechT5, Bark and Massive Multilingual Speech (MMS)

### [7-bark-huggingface-demo.ipynb](./7-bark-huggingface-demo.ipynb)
- Goal: Try advanced TTS with the Bark model on Hugging Face.
- Demonstrated:
  - Generating expressive and controllable synthetic voices.
  - Running interactive speech demos.

---

## Unit 7
### [8-speech-to-speech-translation.ipynb](./8-speech-to-speech-translation.ipynb)
- Goal: Build a speech-to-speech translation pipeline.
- Demonstrated:
  - Combining ASR + translation + TTS.
  - Converting speech in one language to speech in another within a Gradio demo.

### [9-vocal-assistant.ipynb](./9-vocal-assistant.ipynb)
- Goal: Create a simple voice assistant application.
- Demonstrated:
  - Capturing speech input, processing it, and returning responses.
  - Integrating the process in 4 steps:
    1) Wake word detection
    2) Speech Transcription
    3) Language model query
    4) Synthesize speech

### [10-meeting-transcribe.ipynb](./10-meeting-transcribe.ipynb)
- Goal: Transcribe and analyze meeting audio recordings.
- Demonstrated:
  - Multi-speaker transcription and diarization.
  - Structuring transcripts with speaker labels and timestamps.

---

