import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

raw_text = """
a depiction of the greek good Hera in realistic illustrative style as a proud warrior in tropical oasis on greek island
splash combination of spices, garlic, rosemary, lime from the top of simple white and long can , conceptual art , colorful , french space
pale skinned male necromancer with a wicked scar on his face in black robes with pale green swirls of magic coming from hands. fantasy art
indian temple in the middle of a stunning forest, warm blue light, light coming from sky, high definition, ultra realistic, 4k
a storyboard with a frame showing a Honda motorcycle store, large display case displaying motorcycles, street corner, tree. The comic style drawing has thick lines, high contrast and uses black ink and white color.
a dungeons and dragons hydra with female vampire heads HD
a group of druids using magic to create a sacred currency
a storyboard with a frame showing a Honda motorcycle store, large display case displaying motorcycles. The comic style drawing has thick lines, high contrast and uses black ink and white color.
extremely detailed and realistic, grim dark gothic atmosphere, renaissance portrait, total black background, old male halfling, kind granny, playful gaze, dark brown hair, white curly hair, 80 years old, laugghing, wise aze, smile, fairy tribal outfit
extremely detailed and realistic, grim dark gothic atmosphere, renaissance portrait, total black background, old male halfling, kind granny, playful gaze, dark brown hair, white curly hair, 80 years old, laugghing, wise aze, smile
A evil raven dark fantasy fog-covered forest ancient ruins, darkened sky, and flickering shadows, ultra realistic
The cloaked figure walking through an eerie, dark forest. Shadows flicker. low angle of feet. dark fantasy fog-covered forest ancient ruins, darkened sky, and flickering shadows, ultra realistic
A carbon capture facility near the ocean getting multiple tank ships and trailers comming inn to deliver carbon
A cloaked figure reaching out, fingers tracing ancient runes, dark fantasy fog-covered forest ancient ruins, darkened sky, and flickering shadows, ultra realistic
A dark fantasy fog-covered forest or ancient ruins, darkened sky, and flickering shadows, ultra realistic
trastienda de una casa en la narvarte , con una mesa de juego de ajederez, en los muros unas repisas con botellas de quimicos
trastienda traspatio de una casa en la narvarte , con una mesa de juego y un ajederez
extremely detailed and realistic, grim dark gothic atmosphere, renaissance portrait, total black background, dark epic face, elegant and sophisticated warlock, angry, cruel gaze, mad gaze, powerful gaze, multiple ancient amulets, old man, smart and handsome, 60 years old man, dark mage, insane gaze. chaotic gaze, black...
rastienda traspatio de una casa en la narvarte de los 60,estilo californiano, con una mesa de juego ajederez
ultra realistic dark fantasy
extremely detailed and realistic, grim dark gothic atmosphere, renaissance portrait, total black background, dark epic face, elegantt and sophisticated lord, pedant, cruel gaze, cold gaze, proud gaze, multiple amulets, jewellery, old man, smart and handsome, 80 years old man, dark mage, mysterious gaze
trastienda traspatio de una casa en la narvarte de los 60,antigua con un cuervo de mascota
trastienda traspatio de una casa antigua con una mesa de juegos
a photo-realistic, detailed image of a large and thick lettering sign made of bold, sans-serif small-caps letters reading the word "brand" in white, satin-mate plastic, nestled inside a playful confetti blast exploding from the background towards the camera, with colorful serpentines darting out and around it, against ...
trastienda, traspatio de una casa antigua con una mesa de juegos en el centro de la ciudad fotografia chivo lubezky
Edificio deco looby con una recepcion con una fuente mosacos
edificio estilo deco con una recpcion y una fuente
A red panda looking up at the buildings in the city,suprised,by japanese anime style
futuristic battlefield with humanoid robot soldiers in battle. Above them military drones
Isolationia: In the cold and silent realm of Isolationia, the winds bloweth with the sound of naught but thine own breath. The mountains loom high and far, their peaks lost to the gray heavens, and no footstep followeth thine upon the barren earth. Each man is as an isle, untouched and unbothered by his neighbor, for t...
A dazzlingly shimmering, otherworldly sonic symbiote pulsates with energy and light, its form a swirling kaleidoscope of vibrant hues and twisting tendrils. This fantastical being seems to dance and sing in harmony with unseen melodies, its intricate design hinting at a connection to the very fabric of the universe. Th...
A red panda looking up at the buildings in the city,by japanese anime style
photograph row of humanoid robot AI soldiers ready for battle.
A red panda run in the city,a lot of people,by japanese anime style
A cabon capture facility
A red panda in the city,by japanese anime style
realistic photo of adult siblings smiling and hugging with a sunset background
A red panda in the city,a lot of people,by japanese anime style
Little cartoon god Zeus cartoon for kids. . lightning bolt In his hand. Drawn style with pencils
extremely detailed and realistic, grim dark gothic atmosphere, old medieval bottle of red wine and the old ornamented glass of red wine, heraldic emblem, dark sky, grim clouds, grey and purple bottle, black and white bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, huge medieval city...
extremely detailed and realistic, medieval atmosphere, rustic bottle of honey liqueur and medieval glass of honey liqueur, small bees, wild honeycombs on the background, grim sky, poor simple bottle, grey and purple bottle, black and white bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the backgro...
Ultra realistic solar punk
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of spruce vodka and medieval glass of white spruce vodka, spruce branches, grim sky, grey and purple bottle, black and white bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, huge medieval city on the b...
cartoon god Zeus cartoon for kids. Chibi
cocktail spotlit on sidetable in dimly lit post modern home with two people silhouetted moving in the background
hyper-realistic depiction of an Australian Aboriginal woman surrounded by delicately blooming Eucalyptus leaves and flowers in diverse shades of cream. The woman is intricately depicted, with extraordinary detail in her eyes. This vivid background is further embellished by line art that emphasizes the intricate subtlet...
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of potato vodka and medieval glass of white potato vodka, potatoes, grim sky, grey and purple bottle, black and white bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, huge medieval city on the backgrou...
a psychologist in office listening to client on couch who is a humanoid robot
extremely detailed and realistic, medieval atmosphere, old medieval bottle of white wine and the old ornamented glass of white wine, heraldic emblem, sunny day, grey clouds, grey and purple bottle, black and white bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, huge medieval city on...
photograph Synthetic biology in a lab
photograph of robotic soldiers getting ready for battle
extremely detailed and realistic, medieval atmosphere, rustic bottle of apple liqueur and medieval glass of apple liqueur, wild apples, grim sky, poor simple bottle, grey and purple bottle, black and white bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, huge medieval city on the bac...
extremely detailed and realistic, medieval atmosphere, rustic bottle of berry liqueur and medieval glass of liqueur, branch with wild berries, grim sky, poor simple bottle, grey and purple bottle, black and white bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, huge medieval city on ...
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of salt vodka and medieval glass of white salt vodka, salt pieces, grim sky, grey and purple bottle, black and white bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, huge medieval city on the backgroun...
white background room full of datacentres, servers, laptops and file cabbinates clean, illustration, no text
underwater dragon
a full body comic illustrative shot of the greek god hestia in a flamenco outfit with a staff, anime style
A visual representation of a cloud environment, with servers and databases, enveloped by protective layers showcasing encryption, compliance, and vulnerability management. photorealistic
white background orchestra group clean, illustration, no text
dragon tooth big alone black background
perspective photograph of a metal bending machine. Metal fabrication factory. Beauty shot, shallow depth of field, high contrast,
red big dragon egg
In a luminously ethereal grove, a whimsical extraterrestrial fairy with shimmering translucent wings dances amidst floating orbs of light. This enchanting scene is depicted in a vibrant and detailed painting, capturing every intricate detail of the otherworldly creature. The fairy's delicate features are highlighted by...
extremely detailed and realistic, medieval atmosphere, rustic bottle of honey liqueur and medieval glass of honey liqueur, small bees, wild honeycombs on the background, grim sky, poor simple bottle, grey and brown bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce tre...
photograph military communications jamming device mounted on a vehicle
extremely detailed and realistic, medieval atmosphere, rustic bottle of apple liqueur and medieval glass of apple liqueur, wild apples, grim sky, poor simple bottle, grey and brown bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees, brown and grey gammaextremely ...
Animal x Knifefish = Anifefish
a photo-realistic, detailed image of a large and thick lettering sign formed of large, bold, sans-serif small-caps letters reading the word "brand" in white, mate material, emerging from a wild confetti explosion full of colors and serpentines, against a light grayish backdrop.
extremely detailed and realistic, medieval atmosphere, rustic bottle of berry liqueur and medieval glass of liqueur, branch with wild berries, grim sky, poor simple bottle, grey and brown bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees, brown and grey gamma
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of spruce vodka and medieval glass of white spruce vodka, spruce branches, grim sky, grey and brown bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees, brown and grey gamma
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of potato vodka and medieval glass of white potato vodka, potatoes, grim sky, grey and brown bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees, brown and grey gamma
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of salt vodka and medieval glass of white salt vodka, salt pieces, grim sky, grey and brown bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees, brown and grey gamma
photograph image of military logistics, battlefield strategy, and predictive modeling
photograph of Peper Shor and his Shor's algorithm on blackboard behind him
extremely detailed and realistic, medieval atmosphere, old medieval bottle of red wine and the old ornamented glass of red wine, heraldic emblem, beautiful sunset, grey clouds, grey and brown bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees, brown and grey gamm...
extremely detailed and realistic, medieval atmosphere, old medieval bottle of white wine and the old ornamented glass of white wine, heraldic emblem, sunny day, grey clouds, grey and brown bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees, brown and grey gamma
image of quantum encrypted computer
photograph of soldier with AR47 with "Code of War" written in front
a full body illustrative shot of the greek god hestia in a flamenco outfit with a little armour surrounded by flames, anime style
line of autonomous military vehicles waiting for combat
a storyboard with a frame showing a 30 year old latino man leaving his house through the door, looking at the time on his watch, front view of the man, man in the foreground, house with large windows. The comic style drawing has thick lines, high contrast and uses black ink and white color.
Dumb Brute with overall and no shoes. He has an underbite and is absent minded. He is a big human and he is fat. He has no hair or beard. Sitting in the forest holding a bird. Big goofy smile.
extremely detailed and realistic, medieval atmosphere, old medieval bottle of white wine and the old ornamented glass of white wine, heraldic emblem, sunny day, grey clouds, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees, storm
extremely detailed and realistic, medieval atmosphere, rustic bottle of honey liqueur and medieval glass of honey liqueur, small bees, wild honeycombs on the background, grim sky, poor simple bottle, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce tr...
vibrant oil color painting of lord shiva in white background
photograph image of a concert people going in and scanning technology scanning people and doing analysis
photograph AI image recognition technology
simple anime style of very muscular body builder girl in a 80s high rise one piece jazzercise outfit in a dystopian world in the a futuristic style of the 60s clothing wearing headphones and 60’s wedges, ink, comic book cover, loose, ants perspective, dynamic, ink wash,
extremely detailed and realistic, medieval atmosphere, rustic bottle of apple liqueur and medieval glass of apple liqueur, wild apples, grim sky, poor simple bottle, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees
an old medieval solitary with a grape press, in a pencil and charcoal style
extremely detailed and realistic, medieval atmosphere, rustic bottle of berry liqueur and medieval glass of liqueur, branch with wild berries, grim sky, poor simple bottle, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of spruce vodka and medieval glass of white spruce vodka, spruce branches, grim sky, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of potato vodka and medieval glass of white potato vodka, potatoes, grim sky, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees
extremely detailed and realistic, medieval atmosphere, heraldic emblem, old bottle of salt vodka and medieval glass of white salt vodka, salt pieces, grim sky, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees
photograph of futuristic printing device printing artifical limbs in a lab
wearable watch showing heart rate and vital signs
simple anime style of very muscular body builder girl in a 80s high waisted jazzercise outfit in a dystopian world in the a futuristic style of the 60s clothing wearing headphones and 60’s wedges, ink, comic book cover, loose, ants perspective, dynamic, ink wash,
photograph of human hand holding tweezers taking piece of DNA out of DNA sequence
Text field for text Mittelalter style
extremely detailed and realistic, medieval atmosphere, old medieval bottle of red wine and the old ornamented glass of red wine, heraldic emblem, beautiful sunset, grey clouds, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees extremely detailed and realistic, medieval atmosphere, old medieval bottle of red wine and the old ornamented glass of red wine, heraldic emblem, beautiful sunset, grey clouds, grey and purple bottle, cold rocky seashore, pine and spruce trees, dark grim sky on the background, rocks and spruce trees
""".lower().split()

VOCAB_SIZE = 40
word_counts = Counter(raw_text)
vocab_words = ["<UNK>"] + [word for word, count in word_counts.most_common(VOCAB_SIZE - 1)]

word_to_ix = {word: i for i, word in enumerate(vocab_words)}
ix_to_word = {i: word for i, word in enumerate(vocab_words)}

def get_index(word):
    return word_to_ix.get(word, word_to_ix["<UNK>"])

WINDOW_SIZE = 2
data = []
for i in range(WINDOW_SIZE, len(raw_text) - WINDOW_SIZE):
    context = (
        raw_text[i - 2: i] +
        raw_text[i + 1: i + WINDOW_SIZE + 1]
    )
    target = raw_text[i]

    context_idxs = [get_index(w) for w in context]
    target_idx = get_index(target)
    data.append((context_idxs, target_idx))

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        mean_embeds = torch.mean(embeds, dim=1)
        out = self.linear(mean_embeds)
        return out

EMBEDDING_DIM = 20
model = CBOW(len(vocab_words), EMBEDDING_DIM)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 100
for epoch in range(EPOCHS):
    total_loss = 0
    for context, target in data:
        context_tensor = torch.tensor([context], dtype=torch.long)
        target_tensor = torch.tensor([target], dtype=torch.long)

        optimizer.zero_grad()

        log_probs = model(context_tensor)

        loss = loss_function(log_probs, target_tensor)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch: {epoch+1} | Loss: {total_loss/len(data):.4f}")

word_vectors = model.embeddings.weight.data.numpy()

def get_vector(word):
    return word_vectors[get_index(word)]

def find_nearest_neighbors(word, k=5):
    vec = get_vector(word)
    similarities = []
    for i, other_vec in enumerate(word_vectors):
        if ix_to_word[i] == word or ix_to_word[i] == "<UNK>": continue
        sim = np.dot(vec, other_vec) / (np.linalg.norm(vec) * np.linalg.norm(other_vec))
        similarities.append((ix_to_word[i], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def analogy_test(word_a, word_b, word_c):
    vec = get_vector(word_b) - get_vector(word_a) + get_vector(word_c)

    similarities = []
    for i, other_vec in enumerate(word_vectors):
        word = ix_to_word[i]
        if word in [word_a, word_b, word_c, "<UNK>"]: continue
        sim = np.dot(vec, other_vec) / (np.linalg.norm(vec) * np.linalg.norm(other_vec))
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0]

print("\n--- NEAREST NEIGHBORS ---")
print("Nearest to 'dog':", find_nearest_neighbors("dog", 3))
print("Nearest to 'prague':", find_nearest_neighbors("prague", 3))

print("\n--- ANALOGY TEST ---")
ans = analogy_test("man", "woman", "king")
print(f"man : woman = king : {ans[0]} (Similarity: {ans[1]:.2f})")
ans2 = analogy_test("prague", "czechia", "paris")
print(f"prague : czechia = paris : {ans2[0]} (Similarity: {ans2[1]:.2f})")

print("\n--- BIAS TEST ---")
bias_ans = analogy_test("man", "woman", "doctor")
print(f"man : woman = doctor : {bias_ans[0]} (Similarity: {bias_ans[1]:.2f})")

pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(word_vectors)

plt.figure(figsize=(10, 8))
for i, word in enumerate(vocab_words):
    if word == "<UNK>": continue
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), xytext=(5, 2), textcoords='offset points')
plt.title("Word Embeddings PCA Visualization")
plt.grid(True)
plt.show()
