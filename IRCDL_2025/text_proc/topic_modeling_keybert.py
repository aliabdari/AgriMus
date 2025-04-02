from keybert import KeyBERT
import re
import spacy


def clean_transcription(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.strip()


txt = "Learn How to Properly Clean African Violet Leaves"

txt_ = "Hey guys, hope you're all having a wonderful day. " \
       "Some of you have asked how I make my slips to grow sweet potatoes. " \
       "So I thought I'd share that with you today because sweet potatoes and" \
       " potatoes are actually completely different. " \
       "They're not related to each other so the way you grow them is very different. " \
       "Potatoes in a night-shake family, which means it's related to tomatoes, peppers, eggplants. " \
       "So to grow potatoes, you just need to put the entire potato in the ground and those eyes " \
       "would grow new plants. And you'll get more potatoes when you dig in the soil later. " \
       "But with sweet potatoes, it's part of the morning glory family. " \
       "So the way you grow them is you have to basically grow on a completely new plant you " \
       "call the slip out of the sweet potato. Some people just do a cutting from the sweet potato " \
       "vine and in three or four months later, which is how long it takes for sweet potatoes to develop." \
       " When you dig in the soil, you're not going to get any sweet potatoes. " \
       "So here's a sweet potato that I harvest last year. You can see that it's already been sprouting. " \
       "It's definitely needs to be buried and make new plants out of them. Now you can also use " \
       "a sweet potato that hasn't had any sprouts. It would work just as well. So I'm just going " \
       "to cut a piece and put it in soil. Here's some nice fluffy soil. I just put together. " \
       "It's got worm casting, back guano, pearlite, peat moss, compost. You can use coconut corn " \
       "instead of peat moss or any kind of organic potting mix they used to grow your vegetables. " \
       "I generally stay away from things with chicken or cow manure because they were GMO fed. " \
       "So you can pick up some sweet potatoes at your farmer's market or your markets. " \
       "But get the organic ones because the non-organic usually have been coated to prevent them " \
       "from sprouting or you can buy slips instead of making them. Just make sure they " \
       "are grown correctly so you will get sweet potatoes. Another method to make slips is " \
       "to submerge half of your sweet potato in water. But I prefer using the soil method because " \
       "the slips grow a lot faster this way and it's with less effort. In water sometimes they " \
       "would mold so that you would have to change the water. So to grow them in soil all you got to " \
       "do is lightly cover the sweet potato in soil. You don't need to bury it completely and just wait " \
       "for the slips to grow. Here are some slips that have been completely overgrown. I really should have " \
       "removed these and planted them way before. I spent in this pot for I think two or three months. " \
       "As you can see I only have a small piece of the sweet potato in here and make... " \
       "I was able to make all these slips out of that small little piece. You don't need an entire sweet " \
       "potato in there. You can if you want but it's not necessary. These are some sweet potatoes that " \
       "I just dug up a few days ago. I completely missed these for some reason when I was " \
       "harvesting last year. So these are almost a year old in the ground. So you can see that all " \
       "these slips have grown out of this sweet potato. It's got so much roots. These are definitely " \
       "ready to be separated and plant them individually in the ground. If you don't have enough time " \
       "to plant the slips you can just let them sit in water until you're ready to plant them. If some" \
       " of your slips don't have any roots you can let them sit in water for just about a couple of " \
       "weeks you should get some roots growing. By then you can plant them in the ground, a grow bag or " \
       "a large container. This is a spring and summer crop where you harvest right about autumn. " \
       "So you grow them in full sun or at least half a day of sun in good drainage soil so that the " \
       "sweet potatoes wouldn't rot. Unlike regular potatoes sweet potatoes are actually pretty " \
       "straightforward. You just plant them. You don't need to mount them because they grow from the " \
       "roots only. And about three to four months you'll be ready to harvest them. Be sure to check " \
       "out my previous video on how to prepare for the harvest. I'll leave a link below for you. Well " \
       "I hope you guys enjoyed this video. If you did make sure you like it, subscribe or if you have " \
       "any questions for me leave them in the comments. Until then I hope you guys have a wonderful " \
       "week and I'll see you guys in the next video. Bye."

# Initialize KeyBERT model
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

# Extract keywords
topics = kw_model.extract_keywords(clean_transcription(txt), keyphrase_ngram_range=(1, 5), top_n=7)

keywords = [x[0] for x in topics]

nlp = spacy.load("en_core_web_sm")
tokenized_keywords = []
for keyword in keywords:
    doc = nlp(keyword)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    tokenized_keywords.append(set(tokens))

# Find mutual tokens
mutual_tokens = set.intersection(*tokenized_keywords)
union_tokens = set.union(*tokenized_keywords)

print("Mutual Tokens:", mutual_tokens)
print("Union Tokens:", union_tokens)

print("Extracted Topics:", topics)
