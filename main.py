from gnomes_village import papa_gnome
from gnomes_village.papa_gnome import papa_gnome_answers


model, tokenizer = papa_gnome.summon_papa_gnome()

questions = [
    "What is the capital of France?",
    'What is the direction of modern quantum physics?',
    "Write a Fibonacci sequence as python script.",
]


query = 'What is the current time in Miami?' # questions[1]

for chunk in papa_gnome_answers(model, tokenizer, query):
    print(chunk, end="", flush=True)

print()

