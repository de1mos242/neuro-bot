import json

conversations = open("colonel/movie_conversations.txt", 'r', encoding="utf-8").readlines()
lines = open("colonel/movie_lines.txt", 'r', errors="ignore").readlines()



pairs = []

lines_map = {line.split(' ')[0].replace('\n', ''): " ".join(line.split(' ')[8:]).replace('\n','') for line in lines}

for conv in conversations:
    conv_lines = " ".join(conv.split(" ")[6:]).replace("'", '"').replace('\n', '')
    c_lines = json.loads(conv_lines)
    question = c_lines[0]
    answer = c_lines[1]

    pairs.append((question, answer))

print(len(pairs))
print(len(lines_map))

parsed_dialogs = []
for pair in pairs:
    parsed_dialogs.append(lines_map[pair[0]] + '\t' + lines_map[pair[1]] + '\n')

with open("colonel/parsed.txt", 'w') as out:
    out.writelines(parsed_dialogs)
