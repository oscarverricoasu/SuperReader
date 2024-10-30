import ebooklib
from ebooklib import epub
from html.parser import HTMLParser
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

class Parse(HTMLParser): #html parser for epub
    extract = ""

    def handle_data(self, data):
        self.extract += data
        #print(data) #temporary printing for testing

    def get_text(self):
        return self.extract

class readfile:
    def getInput(self): #Only finds files in the same directory right now
        file = input("Input name of file (.txt, .pdf, or .epub) with filetype at end:\n")
        return file

    def checkFile(self, file): #check if file exists
        try:
            with open(file, 'r', encoding='utf-8') as read:
                print("File exists!!!") #temporary for now as this is in progress
        except FileNotFoundError:
            print("Invalid file")

    def readPDF(self, file):
        pdf = pdfium.PdfDocument(file)
        n_pages = len(pdf)
        words = ""
        for i in range(n_pages):
            page = pdf[i]
            textpage = page.get_textpage()
            words += textpage.get_text_bounded()
            #print(words) #temporary printing for testing
        pdf.close()
        return words

    def readEPUB(self, file):
        book = epub.read_epub(file, {'ignore_ncx': True})
        parser = Parse()
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                body = item.get_body_content().decode()
                parser.feed(body) #temporary printing for testing
        words = parser.get_text()
        #print(words) #temporary printing for testing
        return words

    def readTXT(self, file):
        with open(file, encoding="utf-8") as read:
            words = read.read()
            #print(words) #temporary printing for testing
            return words

if __name__ == "__main__": #for testing reading without driver functions
    test = readfile()
    file = test.getInput()
    test.checkFile(file)
    if file.endswith(".pdf"):
        test.readPDF(file)
    elif file.endswith(".epub"):
        test.readEPUB(file)
    elif file.endswith(".txt"):
        test.readTXT(file)
    else: #for unsupported filetypes/random nonsense
        print("Invalid input")
