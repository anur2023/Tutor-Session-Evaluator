import whisper

def transcribe_audio(audio_path: str, model_size: str = "base", output_txt: str = None, verbose: bool = True):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="en")

    if verbose:
        print("Transcription:\n")
        print(result["text"])

    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"\nTranscription saved to {output_txt}")

    return result
