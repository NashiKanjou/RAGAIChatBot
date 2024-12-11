from langchain.docstore.document import Document
import queue

char_break = "\n"

def loadFile(source, chunksize, chunk_overlap):
    q = queue.Queue(maxsize=chunk_overlap)
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
        line_len = len(line)
        if not line:
            break
        else:
            if force:
                if line == char_break or line.isspace() or line.endswith(char_break): #check for new line
                    doc = Document(page_content=lines.strip(),metadata={'source': source})
                    files.append(doc)
                    lines = ""
                    while not q.empty():
                        lines +=q.get()
                    tokens = 0
                    force = False
                    continue
                else:
                    force = True
            if newline:
                newline = False
                if line == char_break or line.isspace(): #check for double new line
                    doc = Document(page_content=lines.strip(),metadata={'source': source})
                    files.append(doc)
                    lines = ""
                    while not q.empty():
                        lines +=q.get()
                    tokens = 0
                    continue
                else:
                    lines += char_break
                    q.put(char_break)
                    tokens += 1
            else:
                if line == char_break or line.isspace():
                    newline = True
                    continue
            if chunksize < tokens + line_len:
                force = True
                lines += line
                for c in line:
                    q.put(c)
                tokens += line_len
                continue
            else:
                lines += line
                for c in line:
                    q.put(c)
                tokens += line_len
    if len(lines) > 0:
        doc = Document(page_content=lines.strip(),metadata={'source': source})
        files.append(doc)
    file.close()
    return files
