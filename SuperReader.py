import spacy
from TTS.api import TTS
from ReadFile import readfile
from names_dataset import NameDataset, NameWrapper
import jsonlines
import random
import time
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load SpaCy Model and TTS
nlp = spacy.load("en_core_web_sm")
try:
    print("Loading TTS model...")
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading TTS model: {e}")

PITCH_FACTOR_RANGE = (0.8, 1.3)

# This will be the main encapsulation for speakers and the superbook structures
class SpeakerManager:

    # Initializer of data holders
    def __init__(self):
        self.speakers = [
            {'name': 'Narrator', 'gender': 'unknown', 'number': 'singular', 'pitch_factor': 1}  # <<< FIXED TYPO
        ]
        self.superbook = []
        self.nd = NameDataset()  # Dataset to guess gender of names that are unknown

    # Adds or updates a speaker's info into the speakers data holder
    def add_speaker(self, name, gender="unknown", number="singular"):
        existing = next((s for s in self.speakers if s["name"] == name), None)
        if existing:
            if existing["gender"] == "unknown" and gender != "unknown":
                existing["gender"] = gender
        else:
            self.speakers.append({
                "name": name,
                "gender": gender,
                "number": number,
                "pitch_factor": random.uniform(*PITCH_FACTOR_RANGE)  # <<< FIXED TYPO
            })

    # Retrieves data of a speaker in the speakers data holder
    def get_speaker(self, name):
        return next((s for s in self.speakers if s["name"] == name), None)

# Generate audiobook files with multithreading
def generate_audio_with_pydub(speaker_manager):
    def process_entry(index, entry, num_digits):
        try:
            speaker_name = entry['speaker']
            text = entry['text']
            speaker_data = speaker_manager.get_speaker(speaker_name)

            if not speaker_data:
                logging.warning(f"No speaker data found for {speaker_name}, skipping.")
                return

            # Generate audio using TTS model
            temp_audio_path = f"audio/temp_{index}.wav"
            output_audio_path = f"audio/{str(index).zfill(num_digits)}_{speaker_name}.wav"

            tts_model.tts_to_file(text=text, file_path=temp_audio_path)

            # Apply pitch shift with pydub
            apply_pitch_shift(temp_audio_path, speaker_data['pitch_factor'], output_audio_path)

            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        except Exception as e:
            logging.error(f"Error processing audio for entry {index}: {e}")

    total_entries = len(speaker_manager.superbook)
    num_digits = len(str(total_entries))
    with ThreadPoolExecutor() as executor:
        for index, entry in enumerate(speaker_manager.superbook, 1):
            executor.submit(process_entry, index, entry, num_digits)

# Modifies a character's line by their unique pitch factor to have a distinct voice using pydub
def apply_pitch_shift(audio_path, pitch_factor, output_path):
    try:
        audio = AudioSegment.from_file(audio_path)

        if pitch_factor != 1:
            altered_audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * pitch_factor)
            }).set_frame_rate(audio.frame_rate)
        else:
            altered_audio = audio

        # Ensure the output audio is normalized in length to avoid mismatches
        fixed_length_audio = altered_audio.set_duration(audio.duration_seconds)

        fixed_length_audio.export(output_path, format="wav")
        print(f"Audio exported with pitch factor: {pitch_factor} to '{output_path}'")

    except Exception as e:
        logging.error(f"Error in pitch shifting for {audio_path}: {e}")

# Function to generate a .json file as output for the program instead of printing results in the console
def save_to_jsonl(speaker_manager, filename):
    """Saves the results to a JSONL file."""
    output_file = f"{os.path.splitext(filename)[0]}.jsonl"
    try:
        with jsonlines.open(output_file, mode="w") as writer:
            for speaker in speaker_manager.speakers:
                writer.write({"speakers": speaker})
            for entry in speaker_manager.superbook:
                writer.write({"superbook": entry})
    except Exception as e:
        logging.error(f"Error saving results: {e}")

# Driver to process the input file
def main():
    start_load = time.time()

    # Check if the 'audio' directory exists, if not, create it
    audio_directory = "./audio"
    if not os.path.exists(audio_directory):
        os.makedirs(audio_directory)
        print(f"Directory '{audio_directory}' created.")
    else:
        print(f"Directory '{audio_directory}' already exists.")

    input_reader = readfile()

    try:
        input_file = input_reader.getInput()
        input_reader.checkFile(input_file)
        if input_file.endswith(".pdf"):
            text = input_reader.readPDF(input_file)
        elif input_file.endswith(".epub"):
            text = input_reader.readEPUB(input_file)
        elif input_file.endswith(".txt"):
            text = input_reader.readTXT(input_file)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        return

    end_load = time.time()

    # Process the text
    start_process = time.time()
    speaker_manager = SpeakerManager()
    lines = text.strip().split("\n")  # Split the text into lines only once # <<< CHANGED LINE
    process_text_lines(lines, speaker_manager)
    end_process = time.time()

    # Query for missing gender for named speakers
    guess_genders_for_speakers(speaker_manager)

    # Generate Audio
    start_audio = time.time()
    generate_audio_with_pydub(speaker_manager)
    end_audio = time.time()

    # Output results
    save_to_jsonl(speaker_manager, input_file)

    # Timers and logging
    logging.info(f"Loading time: {end_load - start_load:.2f} seconds")
    logging.info(f"Processing time: {end_process - start_process:.2f} seconds")
    logging.info(f"Audio generation time: {end_audio - start_audio:.2f} seconds")
    logging.info("Processing complete!")

if __name__ == "__main__":
    main()

# Function to process text lines with improved speaker attribution
def process_text_lines(lines, speaker_manager):
    current_speaker = "Narrator"

    # Check if there are any PERSON entities in the text
    text_has_person_entities = any(
        ent.label_ == "PERSON" for line in lines for ent in nlp(line).ents
    )

    if not text_has_person_entities:
        # If no PERSON entities are found, alternate between unnamed speakers
        alternate_speakers_without_person_entities(lines, speaker_manager)
        return

    for line in lines:
        line_doc = nlp(line)
        named_speaker = get_named_speaker(line_doc)
        gender = infer_gender_from_pronouns(line_doc) if named_speaker else "unknown"

        # Split the line into narration and dialogue
        line_parts = split_narration_dialogue(line)

        # Process each part
        for part in line_parts:
            if part['type'] == 'dialogue':
                if named_speaker:
                    current_speaker = named_speaker
                    speaker_manager.add_speaker(current_speaker, gender)
                speaker_manager.superbook.append({"speaker": current_speaker, 'text': part['text']})
            else:
                narrator_speaker = get_speaker_from_narration(nlp(part['text']))
                speaker_manager.superbook.append({'speaker': 'Narrator', 'text': part['text']})
                if narrator_speaker:
                    speaker_manager.add_speaker(narrator_speaker)

# Alternate between two unnamed speakers
def alternate_speakers_without_person_entities(lines, speaker_manager):
    unnamed_speakers = ['Unnamed Speaker 1', 'Unnamed Speaker 2']
    speaker_manager.add_speaker('Unnamed Speaker 1', "unknown", "singular")
    speaker_manager.add_speaker('Unnamed Speaker 2', "unknown", "singular")

    current_speaker_index = 0  # Tracks which unnamed speaker is currently active

    for line in lines:
        line = line.strip()  # Remove any extra whitespace around the line
        current_speaker = unnamed_speakers[current_speaker_index]
        speaker_manager.superbook.append({'speaker': current_speaker, 'text': line})

        # Switch to the other unnamed speaker
        current_speaker_index = (current_speaker_index + 1) % 2

# Split lines containing both narration and dialogue with regex
def split_narration_dialogue(line):
    pattern = r'([^"]*)(?:"([^"]*)")?'
    matches = re.findall(pattern, line)
    parts = []
    for narration, dialogue in matches:
        if narration.strip():
            parts.append({'type': 'narration', 'text': narration.strip()})
        if dialogue.strip():
            parts.append({'type': 'dialogue', 'text': dialogue.strip()})
    return parts

# Infer gender based on pronouns in the text
def infer_gender_from_pronouns(doc):
    pronoun_map = {
        "he": "male", "him": "male", "his": "male",
        "she": "female", "her": "female", "hers": "female",
        "they": "unknown"
    }
    for token in doc:
        if token.pos_ == "PRON" and token.lower_ in pronoun_map:
            return pronoun_map[token.lower_]
    return "unknown"

# Detect named speaker in dialogue
def get_named_speaker(doc):
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    for token in doc:
        if token.pos_ == "PROPN" and token.dep_ in {"nsubj", "attr"}:
            return token.text
    return None

# Detect speaker from narration
def get_speaker_from_narration(doc):
    for token in doc:
        if token.lemma_ in ['say', 'ask', 'reply', 'shout', 'continue', 'speak', 'add']:
            for child in token.children:
                if child.dep_ == 'nsubj' and (child.ent_type_ == 'PERSON' or child.pos_ in ['PROPN', 'PRON']):
                    return child.text
    return None

# Guess unknown gender for named speakers
def guess_genders_for_speakers(speaker_manager):
    for speaker in speaker_manager.speakers:
        name = speaker.get('name')
        if speaker['gender'] == "unknown" and name != "Narrator":
            try:
                search_result = speaker_manager.nd.search(name)
                if search_result:
                    gender = NameWrapper(search_result).gender.lower()
                    speaker['gender'] = gender
            except Exception as e:
                logging.error(f"Error guessing gender for {name}: {e}")
