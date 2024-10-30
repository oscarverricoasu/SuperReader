import spacy
from ReadFile import readfile
from names_dataset import NameDataset, NameWrapper

# Load SpaCy Model
nlp = spacy.load("en_core_web_sm")

# Initialize Data Holders
current_speaker = "Narrator"    # Set Narrator as the initial speaker
previous_speaker = None         # Keep track of the last valid speaker for dialogue continuity
speakers = []                   # List of detected speakers
superbook = []                  # Output with speakers and their text
nd = NameDataset()              # Dataset to guess gender of names that are unknown

# Add a speaker to the list if they are not already there, with inferred gender if available
def add_speaker(speaker_name, gender="unknown", number="singular"):
    for speaker in speakers:
        if speaker['name'] == speaker_name:
            # Update gender if it’s unknown and we now have new information
            if speaker['gender'] == "unknown" and gender != "unknown":
                speaker['gender'] = gender
            return
    # Add a new speaker if they don’t already exist
    speakers.append({'name': speaker_name, 'gender': gender, 'number': number})

# Function to infer gender based on pronouns in the text
def infer_gender_from_pronouns(doc):
    gender = "unknown"
    for token in doc:
        if token.pos_ == "PRON":
            if token.lower_ in {"he", "him", "his"}:
                return "male"
            elif token.lower_ in {"she", "her", "hers"}:
                return "female"

    return gender

# Function to split lines containing both narration and dialogue
def split_narration_dialogue(line):
    parts = []
    in_dialogue = False
    current_part = []

    for char in line:
        if char == '"':  # Toggle between dialogue and narration
            if in_dialogue:
                parts.append({'type': 'dialogue', 'text': ''.join(current_part).strip()})
                current_part = []
            else:
                if current_part:
                    parts.append({'type': 'narration', 'text': ''.join(current_part).strip()})
                current_part = []
            in_dialogue = not in_dialogue  # Switch between narration/dialogue mode
        else:
            current_part.append(char)

    # Add remaining narration if any after the last quote
    if current_part:
        parts.append({'type': 'narration', 'text': ''.join(current_part).strip()})

    return parts

# Function to detect named speaker in dialogue
def get_named_speaker(input_text):
    for ent in input_text.ents:
        if ent.label_ == "PERSON":
            return ent.text
    for token in input_text:
        if token.pos_ == "PROPN" and token.dep_ in {"nsubj", "attr"}:
            return token.text
    return None

# Function to detect speaker from narration
def get_speaker_from_narration(narration_text):
    doc = nlp(narration_text)
    for token in doc:
        if token.lemma_ in ['say', 'ask', 'reply', 'shout', 'continue', 'speak', 'add']:
            # Find the subject of the speech verb
            subject = None
            for child in token.children:
                if child.dep_ == 'nsubj':
                    if child.ent_type_ == 'PERSON' or child.pos_ in ['PROPN', 'PRON']:
                        subject = child.text
                        break
            if subject:
                return subject
    return None

# New function to alternate between two unnamed speakers
def alternate_speakers_without_person_entities(lines):
    global current_speaker, previous_speaker
    unnamed_speakers = ['Unnamed Speaker 1', 'Unnamed Speaker 2']  # Define the two unnamed speakers
    current_speaker_index = 0  # Track which unnamed speaker is currently active

    for line in lines:
        line = line.strip()  # Remove any extra whitespace around the line

        # Alternate between the unnamed speakers
        current_speaker = unnamed_speakers[current_speaker_index]
        superbook.append({'speaker': current_speaker, 'text': line})

        # Switch the speaker for the next line
        current_speaker_index = (current_speaker_index + 1) % 2

# Function to process text line-by-line with improved speaker attribution
def process_by_lines(lines):
    global current_speaker, previous_speaker
    last_speaker = None  # Keep track of the last speaker
    potential_speakers = []  # List of recent speakers

    # Check if there are any PERSON entities in the text
    text_has_person_entities = any(
        ent.label_ == "PERSON" for line in lines for ent in nlp(line).ents
    )

    if not text_has_person_entities:
        # If no PERSON entities are found, alternate between unnamed speakers
        alternate_speakers_without_person_entities(lines)
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
                    add_speaker(current_speaker, gender=gender)
                    if current_speaker not in potential_speakers:
                        potential_speakers.append(current_speaker)
                else:
                    if len(potential_speakers) >= 2:
                        # Alternate between the two most recent speakers
                        idx = potential_speakers.index(previous_speaker) if previous_speaker in potential_speakers else -1
                        current_speaker = potential_speakers[(idx + 1) % 2]
                    elif previous_speaker:
                        current_speaker = previous_speaker
                    else:
                        current_speaker = "Unknown Speaker"

                superbook.append({'speaker': current_speaker, 'text': part['text']})
                previous_speaker = current_speaker
                last_speaker = current_speaker
            else:
                # Handle narration
                speaker_in_narration = get_speaker_from_narration(part['text'])
                if speaker_in_narration:
                    narration_gender = infer_gender_from_pronouns(nlp(part['text']))
                    add_speaker(speaker_in_narration, gender=narration_gender)
                    superbook.append({'speaker': 'Narrator', 'text': part['text']})
                    previous_speaker = speaker_in_narration
                    if speaker_in_narration not in potential_speakers:
                        potential_speakers.append(speaker_in_narration)
                else:
                    superbook.append({'speaker': 'Narrator', 'text': part['text']})


# Passer to guess unknown gender for named speakers
def guess_genders_for_speakers(speakers):
    for speaker in speakers:
        name = speaker.get('name')
        if name != "Narrator":
            gender = NameWrapper(nd.search(name)).gender.lower()
            add_speaker(name, gender=gender, number=speaker.get('number'))



# Driver to process the input file
if __name__ == "__main__":
    input = readfile()

    file = input.getInput() #do something whileloopish to run until valid input
    input.checkFile(file)  # This needs to actually stop the program

    if file.endswith(".pdf"):
        text = input.readPDF(file)
    elif file.endswith(".epub"):
        text = input.readEPUB(file)
    elif file.endswith(".txt"):
        text = input.readTXT(file)
    else:
        print("Invalid input")
        exit()

    # Process the text
    lines = text.strip().split('\n')
    process_by_lines(lines)

    # Initiate Narrator
    add_speaker('Narrator')

    # Query for missing gender for named speakers
    guess_genders_for_speakers(speakers)

    # Output results
    print("\nSpeakers: ")
    for elem in speakers:
        print(elem)

    print("\nSuperBook: ")
    for elem in superbook:
        print(elem)
