import audio_processing as audio_proc


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
    print(f"\nPhonemes: {sample.phonemes}")
    print(f"Vowels: {sample.vowels}")
    missing = audio_proc.validate_vowel_coverage(sample.vowels)
    if missing:
        print(f"Missing vowels: {" ".join(missing)}")
        msg = "Hmm, that wasn’t clear enough. Please try again, reading the sentence slowly and carefully."
    else:
        normalized = sample.vowels.normalize_z_score(
            *sample.vowels.get_mean_formant_values(),
            *sample.vowels.get_formant_std(),
        )
        normalized.generate_reference_mapping('resources/ref_dict.json')
        msg = "Calibration successful."
    audio_proc.speak_feedback([msg])


if __name__ == "__main__":
    recorder = audio_proc.AudioRecorder()
    transcriber = audio_proc.Transcriber()

    sentence = "He sits near red ants. Dogs saw two blue and grey boats. Her son could hum by the pier."
    user_input = input(
        f"Please read the following sentence to produce a reference of your vowel pronunciation:\n\n"
        f"\"{sentence}\"\n\n"
        f"When you're ready, type R to start recording.\n"
        f"You can type Q at any time to quit.\n\n"
        f"Input: "
    ).strip().upper()
    if user_input == 'Q':
        recorder.destroy()
        exit()
    elif user_input == 'R':
        gen_normalized_ref_vowels(recorder, transcriber)
    else:
        print(f"Input not recognized, please try again.")
