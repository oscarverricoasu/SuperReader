import spacy
import en_core_web_sm


# Load Everything Module
nlp = spacy.load("en_core_web_sm")

# Enter Text Module
text = ''



# Initialize Data Holders
speakers = [] # should be list or set? unsure
dialogue_indicators = {"said", "replied", "asked"}
current_speaker = None

# Process text
doc = nlp(text)

# I don't know whether to switch on sentences or newline characters?
# have to figure out a way to do both and choose based on text?
lines = doc.text.strip().split('\n')


# Function to detect if sentence scanned is a piece of dialogue
def is_dialogue(input_text):
    return any(token.lemma_ in dialogue_indicators for token in input_text)

# Function to detect a named speaker in the sentence
def get_named_speaker(input_text):
    for ent in input_text.ents:
        if ent.label_ == "PERSON":
            return ent.text
        else:
            return None

#function to switch speakers in assumed line switching dialogue
def get_next_speaker(input_speaker, speakers_list):
    if len(speakers_list) == 0:
        return "Speaker 1"
    elif len(speakers_list) == 1 and input_speaker == speakers_list[0]:
        return "Speaker 2"
    elif current_speaker != speakers_list[0]:
        return speakers_list[0]
    else :
        return speakers_list[1]



#newline iterator
for line in lines:
    if is_dialogue(line):
        named_speaker = get_named_speaker(nlp(line))

        if named_speaker:
            current_speaker = named_speaker
            if named_speaker not in speakers:
               speakers.append(named_speaker)

        else:
            current_speaker = get_next_speaker(current_speaker, speakers)

# Sentence iterator
for sentence in doc.sents:
    # Check if dialogue
    if is_dialogue(sentence):
        named_speaker = get_named_speaker(sentence)

        if named_speaker:
            current_speaker = named_speaker
            if named_speaker not in speakers:
                speakers.append(named_speaker)

        else:
            current_speaker = get_next_speaker(current_speaker, speakers)
