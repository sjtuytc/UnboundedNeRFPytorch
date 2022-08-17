import markdown
import pdb

f = open('docs/weekly_nerf.md', 'r')
htmlmarkdown=markdown.markdown( f.read() )

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self, *, convert_charrefs: bool = ...) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        

    def handle_starttag(self, tag, attrs):
        # print("Encountered a start tag:", tag)
        # print(attrs)
        # if len(attrs) < 1 or len(attrs[0]) < 2:
        #     return
        # if attrs[0][0] == 'href':
        #     link = attrs[0][1]
        #     if 'github' not in link:
        #         print(link)
        pass

    def handle_endtag(self, tag):
        # print("Encountered an end tag :", tag)
        pass

    def handle_data(self, data):
        if len(data) < 5 or '2022' in data:
            return
        elif '>' in data:
            pass
            # abstract = data.replace("[code]", "").replace("|", "").replace(">", "").replace("> ", "").strip(" ").replace("\n", "")
            # abstract += "\n"
            # text_file = open("data/abstract.txt", "a")
            # n = text_file.write(abstract)
            # text_file.close()
        else:
            print(data) # print titles here
            # pass

parser = MyHTMLParser()
parser.feed(htmlmarkdown)
pdb.set_trace()