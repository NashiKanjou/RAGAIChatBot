from langchain.docstore.document import Document

char_break = "\n"

def loadFile(source, chunksize):
    files = []
    if not source.endswith(".txt"):
        return files
    file = open(source,"r")
    newline = False
    force = False
    tokens = 0
    lines = ""
    while True:
        line = file.readline()
        line_len = len(line.split(" "))
        if not line:
            break
        else:
            if force:
                if line == char_break: #check for new line
                    doc = Document(page_content=lines.strip(),metadata={'source': source})
                    files.append(doc)
                    lines = ""
                    tokens = 0
                    force = False
                    continue
                else:
                    force = False
            if newline:
                newline = False
                if line == char_break: #check for double new line
                    doc = Document(page_content=lines.strip(),metadata={'source': source})
                    files.append(doc)
                    lines = ""
                    tokens = 0
                    continue
                else:
                    lines += char_break
                    token += 1
            else:
                if line == char_break:
                    newline = True
                    continue
            if chunksize < token + line_len:
                force = True
                lines += line
                tokens += line_len
                continue
            else:
                lines += line
                tokens += line_len
    if len(lines) > 0:
        doc = Document(page_content=lines.strip(),metadata={'source': source})
        files.append(doc)
    file.close()
    return files