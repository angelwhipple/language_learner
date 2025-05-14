from main import speak_feedback
import audio_processing as audio_proc
from pytermgui import tim


def gen_normalized_ref_vowels(recorder, transcriber):
    """
    Normalizes raw formant values F1,F2 for each vowel recorded by the user.
    Saves the normalized values in a reference json file.
    """
    audio = recorder.record(10)
    recorder.playback(audio)
    recorder.save(audio, 'audio/ref.wav')
    transcriber.transcribe('audio/ref.wav', 'audio/ref.lab')
    audio_proc.run_mfa_alignment()
    sample = audio_proc.VocalSample('audio/ref.wav', 'audio/ref.TextGrid')
    phonemes = sample.words[0].phonemes.join_phonemes([word.phonemes for word in sample.words[1:]])
    vowels = sample.words[0].vowels.merge_vowels([word.vowels for word in sample.words[1:]])
    print(f"\nPhonemes: {phonemes}")
    print(f"Vowels: {vowels}")
    missing = audio_proc.validate_vowel_coverage(vowels)
    if missing:
        print(f"Missing vowels: {" ".join(missing)}")
        msg = "Hmm, that wasn’t clear enough. Please try again, reading the sentence slowly and carefully."
    else:
        vowels.gen_reference_data()
        normalized = vowels.normalize_z_score(
            *vowels.get_mean_formant_values(),
            *vowels.get_formant_std(),
        )
        normalized.gen_reference_map('resources/custom_formant_data.json')
        msg = "That was good, thanks!."
    speak_feedback([msg], None)


if __name__ == "__main__":
    recorder = audio_proc.AudioRecorder()
    transcriber = audio_proc.Transcriber()

    sentence = "He sits near red ants. Dogs saw two blue and grey boats. Her son could hum by the pier."
    tim.print(f"Please read the following sentence to produce a reference of your vowel pronunciation:\n\n"
              f"[bold green]{sentence}[/bold green]\n")
    tim.print(f"When you're ready, type [bold]R[/bold] to start recording.\n"
              f"You can type [bold]Q[/bold] at any time to quit.\n")
    user_input = input(f"> ").strip().upper()
    if user_input == 'Q':
        recorder.destroy()
        exit()
    elif user_input == 'R':
        gen_normalized_ref_vowels(recorder, transcriber)
    else:
        print(f"Input not recognized, please try again.")
