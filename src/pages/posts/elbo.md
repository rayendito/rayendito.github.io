---
layout: ../../layouts/post.astro
title: "What is Evidence Lower Bound (ELBO)"
pubDate: 2026-09-21
description: "A writeup about Evidence Lower Bound (ELBO)"
author: "rayendito"
isPinned: false
excerpt: "A short preview for the blog listing."
draft: true
image:
  src: ""
  alt: ""
tags: []
---

tl;dr: diffusion language models often maximize the ELBO, a lower bound on log-likelihood, rather than maximizing log-likelihood directly.

$$
\operatorname{ELBO}(\theta;x)\leq \log p_\theta(x)
$$

## What is a probabilistic model? 

This primer might not be related to what ELBO is yet, but bear with me. Say you want to predict what college major a person is, a probabilistic model can be a probability distribution over the college majors. This would take the form of something like $p(major)$ e.g. $compsci = 0.3$, $\quad law = 0.22$, $\quad humanities = 0.12$ etc.

But this current model $p(major)$ is not very useful because it models everybody the same way. As in, it does not care what a person is like. A more useful model would be the one that also _considers_ what the student is like, that is, $p(major | student)$. This way, the distribution of majors would change depending on the person. This is one definition of a _probabilisitic model_: a model that assigns a distribution over possible outcomes/categories, potentially conditioned also on an input (in this case, what the student is like)

## Now what does that have to do with _language_ models?

What _language models_ do is: Assign probabilities to a given sequence of text $x$. Coming from our previous definition of a _probabilisitic model_, the probability tells us if _$x$ is a plausible sentence or not_. With this being said, sentences that _make sense_ should be assigned higher than gibberish. For example,

> "Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest."

This small sentence (From _The Little Prince_, Antoine de Saint-Exupéry) should be assigned a higher probability, let's say $0.3$ than

>pqwo ususodoi ehrevvevvevvvv llplwokkeoke

(complete nonsense), which should be something like $0.000000002$. So essentially: a _language model_ is a probabilistic model that assigns a probability to a given sentence, which tells you how likely this sentence would occur.

## In Practice

Modern (at least autoregressive ones like ChatGPT) language models assign a probability distribution over a set of vocabulary tokens $\mathcal{V}$, given it's previous tokens.

$$
P(x_t \in \mathcal{V} \mid x_{<t})
$$

Essentially, it gives you what token/word is likely to be next. So for example when we forward pass $x_{<t} = ``\text{the cat sat on the}"$ to the network, we have a probability distribution over $\mathcal{V}$. Likely words are assigned higher probabilities:
$$
P(\text{``mat"} \in \mathcal{V} \mid x_{<t}) = 0.9
$$ 
is higher compared to maybe $P(\text{``mitochondria"} \in \mathcal{V} \mid x_{<t}) = 0.002$. 

At this point, you might be thinking, this is different from what we agreed with? because we agreed that language models assign a probability over whole texts, not predicting what token is next? We are actually still doing the same thing, it's just that probability for whole texts/sentences is now factorized over next token probabilities.

$$
P(x) = 
\underbrace{P(x_1, x_2, \dots, x_n)}_{\text{what we agreed with}} 
= 
\prod_{t=1}^{n}  \underbrace{P(x_t \mid x_1, \dots, x_{t-1})}_{\text{what LMs do in practice}}
$$

## A central challenge in probabilistic models

There are 2 components of probabilistic models (at least what i understood so far)

1. Given a data instance $x$, how likely is this data to occur (what is the probability) according to this model. _How likely is_ $x$, essentially.

2. Generating new $x$ (sampling) from this model.

In the case of our language models, $P(x)$ tells us exactly that. For sampling in language models, we want to generate new sentences. Essentially we do this by running $P(x_t \mid x_1, \dots, x_{t-1})$ and picking the most likely** token over and over again (this is how Autoregressive models work too).

