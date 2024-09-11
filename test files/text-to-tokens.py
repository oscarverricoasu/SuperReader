import spacy
import en_core_web_sm

# This loads the Natural Language Processor in english (hence the "en")
nlp = spacy.load("en_core_web_sm")



input = "The quick brown fox jumped over the lazy dog."

output = nlp(input)

# This will print a list of all scanned words and what part of speech each work is, tokenizing the text.
print("Tokens generated from input text: ")

for token in output:
    print(token.text + " " + token.pos_)


print("\n")


# Named Entity recognition from analyzed text and their assigned label
print("Named Entities recognized in text: ")

if len(output.ents) == 0:
    print("None discovered")
else:
    for ent in output.ents:
        print(ent.text + " " + ent.label_)