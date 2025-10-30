def chunk_text(text, chunk_size=300, overlap=0):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())

        start = end - overlap

        if start < 0:
            start = 0

    return chunks

with open("cleaned_output.txt", "r", encoding="utf-8") as f:
    data = f.read()

chunks_300 = chunk_text(data, 300, 0)
chunks_400 = chunk_text(data, 400, 0)
chunks_500 = chunk_text(data, 500, 0)

chunks_300_ol = chunk_text(data, 300, 100)

print(len(chunks_500[0]))
print(len(chunks_500[1]))
print(len(chunks_500[2]))
print(len(chunks_500[3]))
print(len(chunks_500[4]))
print(chunks_500[4])
print(chunks_300_ol[0])
print(chunks_300_ol[1])



