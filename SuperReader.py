import spacy
import en_core_web_sm
from ReadFile import readfile

# Load SpaCy Module
nlp = spacy.load("en_core_web_sm")

# Load Text Module
text = ''

# Initialize Data Holders
speakers = []                                       # List of speakers in a text
dialogue_indicators = {"said", "replied", "asked"}  # List of dialogue indicators of speaking in text
current_speaker = None                              # Pointer to the current speaker of a text
recent_speakers = []                                # Buffer to keep most recent speakers of a text
buffer_size = 3                                     # Max size of the recent speakers list

superbook = []                                      # MAIN STRUCTURE of speakers and lines in order

# # # HELPER FUNCTIONS # # #

# Function to detect if sentence scanned is a piece of dialogue
def is_dialogue(input_text):
    print("Used is_dialogue")  # debug flag
    return any(token.lemma_ in dialogue_indicators for token in input_text)

# Function to add speaker with gender and number information
def add_speaker(speaker_name, gender="unknown", number="singular"):
    print("Used add_speaker")  # debug flag
    if not any(speaker['name'] == speaker_name for speaker in speakers):
        speakers.append({'name': speaker_name, 'gender': gender, 'number': number})
        print(f"Speaker list now: {speakers}")  # debug flag

# Function to create a memory of recent speakers in a context
def add_to_context_buffer(speaker, sentence_text, gender=None, number="singular", buffer_size=3):
    print(f"Used add_to_context_buffer with speaker: {speaker}")  # debug flag
    recent_speakers.append({'name': speaker, 'text': sentence_text, 'gender': gender, 'number': number})
    if len(recent_speakers) > buffer_size:
        recent_speakers.pop(0)

# Function to detect a named speaker in the sentence
def get_named_speaker(input_text):
    print("Used get_named_speaker")  # debug flag
    for ent in input_text.ents:
        if ent.label_ == "PERSON":
            return ent.text

    # If no named entity, check for pronouns
    for token in input_text:
        if token.pos_ == "PRON" and token.text.lower() in {"he", "she", "they"}:
            gender, number = get_pronoun_speaker(token.text)
            # Use recent_speakers for context-based speaking
            if len(recent_speakers) > 0:
                for prev_speaker in reversed(recent_speakers):
                    if prev_speaker['gender'] == gender and prev_speaker['number'] == number:
                        return prev_speaker['name']
            return "Pronoun-Speaker"  # If no match, return pronoun-based speaker
    return "Narrator"  # Default to Narrator

# Function to detect gender and plurality from pronouns
def get_pronoun_speaker(pronoun):
    print("Used get_pronoun_speaker")  # debug flag
    if pronoun.lower() == "he":
        return "male", "singular"
    elif pronoun.lower() == "she":
        return "female", "singular"
    elif pronoun.lower() == "they":
        return "unknown", "plural"
    return None, None

# Function to switch speakers in assumed line-switching dialogue
def get_next_speaker(input_speaker, speakers_list):
    print("Used get_next_speaker")  # debug flag
    if len(speakers_list) == 0:
        return "Speaker 1"
    elif len(speakers_list) == 1:
        return "Speaker 2" if input_speaker == speakers_list[0] else speakers_list[0]
    else:
        if input_speaker in speakers_list:
            idx = (speakers_list.index(input_speaker) + 1) % len(speakers_list)
            return speakers_list[idx]
        else:
            return speakers_list[0]  # Default to the first speaker

# Function to pick how a document gets divided to associate with speakers
def pick_process(document):
    print("Used pick_process")  # debug flag
    lines = document.text.strip().split('\n')
    if len(lines) < len(list(document.sents)):
        process_by_lines(lines)
    else:
        process_by_sentences(document)

# Newline Iterator Analysis Process
def process_by_lines(lines):
    print("Used process_by_lines")  # debug flag
    global current_speaker
    for line in lines:
        line_doc = nlp(line)
        if is_dialogue(line_doc):
            named_speaker = get_named_speaker(line_doc)
            if named_speaker == "Pronoun-Speaker":
                named_speaker = current_speaker
            elif named_speaker == "Dialogue-Indicated":
                named_speaker = get_next_speaker(current_speaker, speakers)

            if named_speaker and named_speaker != current_speaker:
                current_speaker = named_speaker
                if named_speaker not in speakers:
                    add_speaker(named_speaker)

            add_to_context_buffer(current_speaker, line)
        else:  # General narration case
            print("No dialogue detected")  # debug flag
            current_speaker = "Narrator"
            add_speaker("Narrator")
            add_to_context_buffer(current_speaker, line)

        superbook.append({'speaker': current_speaker, 'text': line})
# Sentence Iterator Analysis Process
def process_by_sentences(document):
    print("Used process_by_sentences")  # debug flag
    global current_speaker
    for sentence in document.sents:
        if is_dialogue(sentence):
            named_speaker = get_named_speaker(sentence)
            if named_speaker == "Pronoun-Speaker":
                named_speaker = current_speaker
            elif named_speaker == "Dialogue-Indicated":
                named_speaker = get_next_speaker(current_speaker, speakers)

            if named_speaker and named_speaker != current_speaker:
                current_speaker = named_speaker
                if named_speaker not in speakers:
                    add_speaker(named_speaker)

            add_to_context_buffer(current_speaker, sentence)
        else:  # General narration case
            print("No dialogue detected")  # debug flag
            current_speaker = "Narrator"
            add_speaker("Narrator")
            add_to_context_buffer(current_speaker, sentence)

        superbook.append({'speaker': current_speaker, 'text':sentence.text})
# # # DRIVER FOR PROGRAM # # #

if __name__ == "__main__":
    # Get the text to be processed
    input = readfile()
    file = input.getInput() #do something whileloopish to run until valid input
    input.checkFile(file)  # This needs to actually stop the program

    if file.endswith(".pdf"):
        text = input.readPDF(file)
    elif file.endswith(".epub"):
        text = input.readEPUB(file)
    elif file.endswith(".txt"):
        text = input.readTXT(file)
    else:  # For unsupported file types/random nonsense
        print("Invalid input")

    # Process the text
    doc = nlp(text)

    # Determine structure of text
    pick_process(doc)

    # Debug for values stored in dataset
    print("Speakers: ", speakers)
    print("SuperBook: ", superbook)
