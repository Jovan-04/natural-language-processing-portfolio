lesson plan:
Synsets (defns) -> Lemmas (words) are part of those -> what other relationships do we have? -> semantic similarity

* ~10 mins until first group exercise (slow down...)  
* ~5 mins for exercise 1
* 

I thought it was better to start from the concept of a synset because that's what WordNet is centered around. That might be a little confusing, so I should pause and make sure that made sense to everyone.

Ask for class input with `bank`'s definitions:
- place to deposit a check
- riverbank
- piggy bank
- bank of switches

antonymy - doesn't always make sense on a conceptual level 
I would guess that it stems from the fact that lemmas are defined at a more specific level within synsets. Technically, `hot` is represented as many different lemmas, depending on which synset/definition you're looking at (notice how our `jump` and `leap` lemmas from earlier are both within the `jump.n.01` synset).
Based on how we actually use words in English, not everything within `hot` (meaning high temperature) has an opposite - 'hot' & 'cold' are antonyms, but what about 'scalding'? 

What are all the relations that WordNet keeps track of?
- synonyms (synsets)
- antonyms
- hyper/hyponyms (specificity of a concept)
- entailment (requirement; X requires Y)
- meronym/holonym (part/whole)
- pertainym
- troponym
- more?

how exactly does WordNet calculate similarity?
1 / (dist + 1)
    - very few values close to 1; similarity scores fall off quickly
    - can we rescale that somehow?
2k / (i + j + 2k)
    - large k means there's a bias towards 1
    - small k means there's a bias towards 0 / (i + j)

semantic similarity group exercise
- embeddings; categories were broad enough that you'll probably have lots of unrelated words in related headline titles
- WordNet; it's centered around synonym sets, so it can suggest other things within the same synset
- WordNet; with a medical encyclopedia, I think being understand how it got to the results it did is important
- embeddings; too broad for WordNet

creative synthesis ideas

