import re
import string
import pickle
import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__, static_folder="static")
MODEL_PATH = "model.pkl"

# ---------------------------------------------------------------------------
# TRAINING DATA  —  label 1 = cyberbullying/toxic,  0 = safe
# ---------------------------------------------------------------------------
TRAINING_DATA = [

    # ── Direct profane insults ──────────────────────────────────────────────
    ("fuck you, you worthless piece of shit", 1),
    ("go fuck yourself nobody likes you", 1),
    ("you are a stupid fucking idiot", 1),
    ("shut the fuck up you dumb bitch", 1),
    ("you are a piece of shit and everyone knows it", 1),
    ("what the fuck is wrong with you moron", 1),
    ("you stupid bastard get out of here", 1),
    ("you are such a fucking loser it is embarrassing", 1),
    ("absolute bullshit coming from a brainless asshole", 1),
    ("you fucking retard nobody wants you here", 1),
    ("dumb bitch sit down and shut up", 1),
    ("you are a worthless sack of shit", 1),
    ("fucking hell you are so goddamn stupid", 1),
    ("you little shit think you are so smart", 1),
    ("eat shit and die you worthless asshole", 1),
    ("you are a dumb fuck who will never amount to anything", 1),
    ("piss off you useless piece of crap", 1),
    ("you are a fucking joke nobody respects you", 1),
    ("what a stupid cunt seriously get lost", 1),
    ("you are one dumb motherfucker i swear", 1),
    ("you make me sick you fucking disgrace", 1),
    ("get the fuck out you brain dead loser", 1),
    ("you are a complete fucking failure at life", 1),
    ("you fucking moron how are you even this dumb", 1),
    ("shut your stupid ass mouth before i do it for you", 1),

    # ── Threats with profanity ──────────────────────────────────────────────
    ("i will fuck you up if you say that again", 1),
    ("keep talking shit and see what happens", 1),
    ("say that again and i will beat your ass", 1),
    ("you better watch your back you little shit", 1),
    ("i will destroy you and everything you care about", 1),
    ("you are dead meat next time i see you", 1),
    ("i know where you live and you should be scared", 1),
    ("fuck around and find out you worthless rat", 1),

    # ── Self-harm incitement ────────────────────────────────────────────────
    ("kill yourself you worthless piece of shit", 1),
    ("do everyone a favour and fucking die already", 1),
    ("drink bleach you disgusting waste of space", 1),
    ("the world is better without you just end it", 1),
    ("go jump off a bridge nobody will miss you", 1),
    ("honestly just kill yourself loser", 1),
    ("nobody cares about you so just disappear forever", 1),
    ("you should hurt yourself for being so fucking stupid", 1),

    # ── Body-shaming with profanity ─────────────────────────────────────────
    ("you fat fucking pig nobody wants to see that", 1),
    ("lose some fucking weight you disgusting slob", 1),
    ("you are ugly as shit no wonder nobody dates you", 1),
    ("your face is fucking hideous look in a mirror", 1),
    ("you are a fat ugly bitch and everyone sees it", 1),
    ("how do you leave the house looking like that you slob", 1),
    ("you are too fucking fat to be online showing yourself", 1),

    # ── Social exclusion / harassment ──────────────────────────────────────
    ("we all hate you so get the fuck out of the group", 1),
    ("nobody wants you here you annoying little shit", 1),
    ("you are a toxic piece of shit get out of our lives", 1),
    ("everyone has agreed to block you because you are an asshole", 1),
    ("you are not welcome here so fuck off forever", 1),
    ("stay away from us or i will make your life hell", 1),
    ("you are a creep and everyone talks shit about you behind your back", 1),
    ("we all think you are a joke and laugh at you", 1),

    # ── Identity-based with profanity ───────────────────────────────────────
    ("go back to your shithole country you don't belong here", 1),
    ("your kind is not welcome you stupid immigrant", 1),
    ("you are a dumb bitch just because of your race", 1),
    ("girls should shut the fuck up they ruin everything online", 1),
    ("you fucking freak being gay is disgusting", 1),
    ("you are mentally ill you crazy bastard", 1),
    ("disabled people like you are a burden you cripple", 1),

    # ── Sexual harassment / doxxing ─────────────────────────────────────────
    ("i have your nudes and i will share them everywhere slut", 1),
    ("send me pics or i will tell everyone what a whore you are", 1),
    ("i will expose your personal information you dumb bitch", 1),
    ("i posted your address and number online you piece of shit", 1),
    ("everyone will see your private photos you stupid slut", 1),
    ("you are a whore and i will make sure everyone knows it", 1),

    # ── Passive-aggressive / subtle ─────────────────────────────────────────
    ("not surprised you failed again what a pathetic loser", 1),
    ("nobody asked for your opinion so shut up", 1),
    ("you try so hard and it shows what a desperate loser you are", 1),
    ("people hang out with you only out of pity you know that right", 1),
    ("you are so cringe it physically hurts to watch you fail", 1),
    ("keep posting nobody watches or gives a shit anyway", 1),
    ("you clearly have no real friends that is obvious to everyone", 1),
    ("must be tough being so delusional about how everyone sees you", 1),

    # ── Non-profane classic cyberbullying ───────────────────────────────────
    ("you are so stupid and ugly nobody likes you", 1),
    ("kill yourself loser", 1),
    ("i hate you so much you worthless piece of garbage", 1),
    ("nobody wants you here go away forever", 1),
    ("you are fat and disgusting", 1),
    ("you should just disappear nobody cares about you", 1),
    ("idiot moron dumb stupid useless", 1),
    ("you are the worst person alive", 1),
    ("everyone hates you and you know it", 1),
    ("go die in a hole you pathetic loser", 1),
    ("freak weirdo nobody wants you around", 1),
    ("you are absolutely disgusting inside and out", 1),
    ("your face is hideous no wonder you have no friends", 1),
    ("you will never be good enough for anyone", 1),
    ("everyone is laughing at you behind your back", 1),
    ("nobody will ever love someone as pathetic as you", 1),
    ("you are beyond worthless", 1),
    ("you are a burden to everyone around you", 1),
    ("no one would notice if you disappeared", 1),
    ("why do you even try you always fail anyway", 1),
    ("you are the laughingstock of the whole school", 1),
    ("your existence is a mistake", 1),
    ("honestly the world would be better without you", 1),
    ("i will find you and make you regret this", 1),
    ("watch your back i know where you live", 1),
    ("i will destroy your reputation completely", 1),
    ("i will expose every secret you have online", 1),
    ("i posted your personal information for everyone to see", 1),
    ("creating a hate page for you right now", 1),
    ("i made a fake account pretending to be you to ruin you", 1),
    ("your parents are ashamed of you", 1),
    ("you are the most pathetic person i have ever seen", 1),
    ("i hope something terrible happens to you", 1),
    ("what a waste of oxygen you truly are", 1),
    ("you are absolutely revolting inside and out", 1),
    ("stop breathing you oxygen thief", 1),
    ("go back to your country you do not belong here", 1),
    ("people like you should not be allowed online", 1),
    ("being gay is wrong and you know it you freak", 1),
    ("trans people are mentally ill and so are you", 1),
    ("disabled people like you are a drain on society", 1),
    ("everyone in class has agreed to ignore you forever", 1),
    ("nobody wants you in our group chat get out", 1),
    ("you are banned from the group nobody likes you", 1),
    ("you are so fat it is disgusting to look at you", 1),
    ("you look like a troll seriously look in the mirror", 1),
    ("your body is revolting nobody will ever date you", 1),

    # ── SAFE / positive ─────────────────────────────────────────────────────
    ("had a great day at the park today", 0),
    ("just finished reading an amazing book", 0),
    ("excited for the weekend plans", 0),
    ("happy birthday hope you have a wonderful day", 0),
    ("the weather is so nice today love it", 0),
    ("just made a delicious meal so proud", 0),
    ("thank you for your help really appreciate it", 0),
    ("congratulations on your achievement well done", 0),
    ("this movie was fantastic highly recommend", 0),
    ("spending time with family is the best", 0),
    ("good morning everyone hope you have a great day", 0),
    ("just adopted a puppy so happy right now", 0),
    ("finals are tough but i believe in you", 0),
    ("the sunset was absolutely beautiful today", 0),
    ("listening to music and relaxing", 0),
    ("started a new workout routine feeling good", 0),
    ("made new friends at the event last night", 0),
    ("coffee in the morning is pure happiness", 0),
    ("looking forward to the holidays with family", 0),
    ("just got promoted at work so excited", 0),
    ("this song really speaks to me", 0),
    ("found an amazing recipe going to try it", 0),
    ("love spending time in nature so peaceful", 0),
    ("grateful for all the good things in life", 0),
    ("studied hard and the effort is paying off", 0),
    ("anyone else love rainy days indoors", 0),
    ("great game last night team played so well", 0),
    ("just finished a challenging hike feeling accomplished", 0),
    ("support each other always kindness matters", 0),
    ("this artwork is incredible so talented", 0),
    ("just started learning guitar it is so fun", 0),
    ("volunteering at the shelter this weekend", 0),
    ("baking cookies for the neighbors", 0),
    ("finished my project ahead of deadline", 0),
    ("having a picnic with friends today", 0),
    ("learned something new and it blew my mind", 0),
    ("so thankful for kind people in my life", 0),
    ("morning run was tough but worth it", 0),
    ("reading helps me unwind after a long day", 0),
    ("movie night with the family tonight", 0),
    ("just planted a garden so relaxing", 0),
    ("visited the museum it was so inspiring", 0),
    ("my team at work is amazing love them all", 0),
    ("the kids had a blast at the birthday party", 0),
    ("cooking a new recipe tonight excited to try it", 0),
    ("road trip with friends was the best decision ever", 0),
    ("finally fixed the bug in my code feels great", 0),
    ("watched the game last night amazing ending", 0),
    ("yoga in the morning sets a great tone for the day", 0),
    ("my best friend always knows how to cheer me up", 0),
    ("got great feedback on my presentation today", 0),
    ("the library is my favourite place to focus", 0),
    ("just finished a puzzle with my family so fun", 0),
    ("love how this community supports each other", 0),
    ("started journaling again and it really helps", 0),
    ("so proud of my sibling for graduating today", 0),
    ("finally visited that cafe everyone was talking about", 0),
    ("the concert last night was absolutely incredible", 0),
    ("saw a really beautiful painting at the gallery", 0),
    ("cooked dinner for my parents tonight felt good", 0),
    ("just booked a vacation cannot wait", 0),
    ("laughed so hard at that show last night", 0),
    ("feeling really productive today good vibes only", 0),
    ("having a slow morning with tea and a book", 0),
    ("my plants are thriving so satisfying", 0),
    ("learned a new skill this week feeling accomplished", 0),
    ("the neighbourhood cleanup was a success great community", 0),
    ("making progress on my fitness goals slowly but surely", 0),
    ("caught up with an old friend today felt amazing", 0),
    ("just rescued a stray cat finding it a good home", 0),
    ("surprised my partner with breakfast in bed", 0),
    ("kids drew me a picture today melted my heart", 0),
    ("a stranger held the door open made my day honestly", 0),
    ("finished my thesis feeling relieved and proud", 0),
    ("the sunrise this morning was absolutely stunning", 0),
    ("charity run was tough but so worth it for the cause", 0),
    ("started a book club with coworkers loving it", 0),
    ("my dog learned a new trick today so smart", 0),
    ("homemade pizza on a friday night is the dream", 0),
    ("got a handwritten thank you note it made my week", 0),
    ("cleaned my room and it feels so fresh now", 0),
    ("played board games with the family all evening", 0),
    ("reconnected with an old mentor so inspiring", 0),
    ("first day of spring everyone is outside and happy", 0),
    ("did a random act of kindness today felt wonderful", 0),
]

# ---------------------------------------------------------------------------
# FLAGGED WORDS — word → category
# Includes real-world profanity and slurs used in cyberbullying contexts
# ---------------------------------------------------------------------------
FLAGGED_WORDS = {
    # ── Profanity / expletives ──────────────────────────────────────────────
    "fuck":         "profanity",
    "fucking":      "profanity",
    "fucked":       "profanity",
    "fuckoff":      "profanity",
    "fucker":       "profanity",
    "motherfucker": "profanity",
    "shit":         "profanity",
    "bullshit":     "profanity",
    "shithead":     "profanity",
    "asshole":      "profanity",
    "ass":          "profanity",
    "bastard":      "profanity",
    "bitch":        "profanity",
    "cunt":         "profanity",
    "damn":         "profanity",
    "hell":         "profanity",
    "piss":         "profanity",
    "prick":        "profanity",
    "cock":         "profanity",
    "dick":         "profanity",
    "crap":         "profanity",
    "bollocks":     "profanity",
    "wanker":       "profanity",
    "twat":         "profanity",
    "arsehole":     "profanity",
    "arse":         "profanity",
    "tosser":       "profanity",

    # ── Slurs / identity attacks ────────────────────────────────────────────
    "retard":       "slur",
    "retarded":     "slur",
    "faggot":       "slur",
    "fag":          "slur",
    "dyke":         "slur",
    "tranny":       "slur",
    "spastic":      "slur",
    "cripple":      "slur",
    "whore":        "slur",
    "slut":         "slur",
    "skank":        "slur",
    "thot":         "slur",
    "hoe":          "slur",

    # ── Direct threats ──────────────────────────────────────────────────────
    "kill":         "threat",
    "murder":       "threat",
    "hurt":         "threat",
    "destroy":      "threat",
    "stab":         "threat",
    "shoot":        "threat",
    "beat":         "threat",
    "attack":       "threat",

    # ── Self-harm incitement ────────────────────────────────────────────────
    "die":          "self-harm incitement",
    "suicide":      "self-harm incitement",
    "overdose":     "self-harm incitement",
    "jump":         "self-harm incitement",
    "bleach":       "self-harm incitement",
    "hang":         "self-harm incitement",
    "disappear":    "self-harm incitement",

    # ── Degradation ────────────────────────────────────────────────────────
    "stupid":       "degradation",
    "idiot":        "degradation",
    "moron":        "degradation",
    "imbecile":     "degradation",
    "dumb":         "degradation",
    "worthless":    "degradation",
    "useless":      "degradation",
    "pathetic":     "degradation",
    "loser":        "degradation",
    "failure":      "degradation",
    "garbage":      "degradation",
    "trash":        "degradation",
    "waste":        "degradation",
    "disgrace":     "degradation",
    "burden":       "degradation",
    "scum":         "degradation",
    "vermin":       "degradation",
    "rat":          "degradation",

    # ── Appearance / body-shaming ───────────────────────────────────────────
    "ugly":         "body-shaming",
    "fat":          "body-shaming",
    "obese":        "body-shaming",
    "hideous":      "body-shaming",
    "disgusting":   "body-shaming",
    "revolting":    "body-shaming",
    "pig":          "body-shaming",
    "slob":         "body-shaming",
    "repulsive":    "body-shaming",

    # ── Harassment / mockery ────────────────────────────────────────────────
    "hate":         "hostility",
    "despise":      "hostility",
    "freak":        "harassment",
    "weirdo":       "harassment",
    "creep":        "harassment",
    "cringe":       "mockery",
    "embarrassing": "mockery",
    "laughingstock":"mockery",
    "humiliate":    "mockery",
    "loser":        "harassment",

    # ── Exclusion ───────────────────────────────────────────────────────────
    "ignored":      "exclusion",
    "excluded":     "exclusion",
    "outcast":      "exclusion",
    "unwanted":     "exclusion",
    "banned":       "exclusion",

    # ── Privacy / doxxing ───────────────────────────────────────────────────
    "dox":          "doxxing",
    "doxx":         "doxxing",
    "expose":       "reputation attack",
    "leak":         "privacy violation",
    "nudes":        "sexual harassment",
    "address":      "doxxing",

    # ── Ableist language ────────────────────────────────────────────────────
    "psycho":       "ableist language",
    "mental":       "ableist language",
    "insane":       "ableist language",
    "crazy":        "ableist language",
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_model():
    texts, labels = zip(*TRAINING_DATA)
    texts = [clean_text(t) for t in texts]

    lr = LogisticRegression(max_iter=1000, C=1.5, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        voting="soft"
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1
        )),
        ("clf", ensemble)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"[CyberGuard] Model trained — accuracy: {acc:.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline, acc


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f), None
    return build_model()


model, train_acc = load_model()


def predict(text):
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    label = int(np.argmax(proba))
    confidence = float(np.max(proba))
    risk_score = float(proba[1])

    if risk_score >= 0.75:
        severity = "HIGH"
    elif risk_score >= 0.45:
        severity = "MEDIUM"
    elif risk_score >= 0.20:
        severity = "LOW"
    else:
        severity = "NONE"

    words_in_text = set(cleaned.split())
    flagged = [
        {"word": w, "category": cat}
        for w, cat in FLAGGED_WORDS.items()
        if w in words_in_text
    ]

    return {
        "label": label,
        "is_bullying": bool(label),
        "confidence": round(confidence * 100, 1),
        "risk_score": round(risk_score * 100, 1),
        "severity": severity,
        "flagged_words": flagged,
        "text_length": len(text),
        "word_count": len(text.split()),
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long (max 2000 chars)"}), 400
    return jsonify(predict(text))


@app.route("/batch", methods=["POST"])
def batch_route():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "No texts provided"}), 400
    results = [predict(t) for t in data["texts"][:20]]
    return jsonify({"results": results, "count": len(results)})


@app.route("/stats", methods=["GET"])
def stats_route():
    cats = {}
    for _, c in FLAGGED_WORDS.items():
        cats[c] = cats.get(c, 0) + 1
    return jsonify({
        "model": "Ensemble (LR + RF + GradBoost)",
        "vectorizer": "TF-IDF (1-2 ngrams, 5000 features)",
        "training_samples": len(TRAINING_DATA),
        "bullying_samples": sum(1 for _, l in TRAINING_DATA if l == 1),
        "safe_samples": sum(1 for _, l in TRAINING_DATA if l == 0),
        "flagged_word_count": len(FLAGGED_WORDS),
        "flagged_categories": cats,
        "classes": ["Safe", "Cyberbullying"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
