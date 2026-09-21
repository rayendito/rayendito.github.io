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

This writeup is a walkthrough of what it is, and mostly why does it solve an intractable problem in minimizing the loss of diffusion language models.

## What is a probabilistic model? 

This primer might not be related to what ELBO is yet, but bear with me. Say you want to predict what college major a person is, a probabilistic model can be a probability distribution over the college majors. This would take the form of something like $p(major)$ e.g. $compsci = 0.3$, $\quad law = 0.22$, $\quad humanities = 0.12$ etc.

But this current model $p(major)$ is not very useful because it models everybody the same way. As in, it does not care what a person is like. A more useful model would be the one that also _considers_ what the student is like, that is, $p(major | student)$. This way, the distribution of majors would change depending on the person. This is one definition of a _probabilisitic model_: a model that assigns a distribution over possible outcomes/categories, potentially conditioned also on an input (in this case, what the student is like)

## Now what does that have to do with _language_ models?

What _language models_ do is: Assign probabilities to a given sequence of text $x$. Coming from our previous definition of a _probabilisitic model_, the probability tells us if _$x$ is a plausible sentence or not_. With this being said, sentences that _make sense_ should be assigned higher than gibberish. For example,

> Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest.

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

There are 2 things you can do with probabilistic models

1. Given a data instance $x$, we can get how the probability of this data is to occur. _How likely is_ $x$, essentially.

2. Generating new $x$ (sampling) from this model. In the case of language models, we are _generating_ new sentences from the model.

In the case of our language models, $P(x)$ tells us exactly that. For sampling, we run $P(x_t \mid x_1, \dots, x_{t-1})$ and pick the most likely\footnote{This is called greedy decoding, i believe there is a whole research direction in language model sampling} token over and over again (this is how Autoregressive models work too).

_Luckily_, doing both is easy for language models. As has been discussed before, we can estimate a sentence $x$ by just forwarding the sentence through the network, or running the forward over and over again to sample/_generate a new sentence_ from the model. I think, or rather i _believe_ not all probabilistic models are easy in both\footnote{i am pretty sure}.

## Many Ways to Evaluate P($x$)
Circling back to our college major example with the distribution of $P_{major}$, _evaluating_ the probability of college major, let's say "medicine" can look like

$$
P_{major}(\text{medicine}) = \frac{\#(\text{students in medicine})}{\text{total students}}
$$

Which gives us the probability of, say picking a random person and them being in _medicine_. But of course, like many other things in life, language models are not that simple. How we estimate the probability of a text via a language model is to .forward() the neural network, which is a lot more complicated than just a lookup table of frequencies\footnote{although early language models do, look up n-gram language models}. I _think_ the reason why we moved on from only using frequencies is because we're not gonna be able to assign an appropriate probability value to a piece of text just because it did not show up _verbatim_ in the training data\footnote{although there are ways to do so, but of course, they are limited} eventhough the text makes sense to a human reader. We want a model that is _rich_, in the sense that it can generalize to sentences it hasn't seen during training.

## Latent Variables

We mentioned earlier that it's better to _predict_ the college major of a person if we also include the properties of the student. Let's say $\text{student} = \{\text{courses taken},\; \text{desk location},\; \text{supervisor}\}$. We'll have a better model if we do

$$
P(major | \{\text{courses taken},\; \text{desk location},\; \text{supervisor}\})
$$

We call the $\text{student}$ information a _latent_ variable that will make our college major estimation better. We can apply the same concept to language models. Let's say we have the same sentence as before (shout out Antoine once again):

> Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest.

This time, we have a _latent variable_ for this sentence, we call

$$
\text{sentence\_attribute} =  \{\text{genre},\; \text{year},\; \text{audience},\; \text{country}\}
$$

With $\text{sentence\_attribute}$ defined like this, we can have multiple _instances_ of this set of attribute that are possible explanations of the text. One instance could be 

$$
\text{sentence\_attribute}_1 =  \{\text{fantasy},\; \text{1990-2000},\; \text{children},\; \text{france}\}
$$

But $\text{sentence\_attribute}_1$ is not the only _plausible_ set of attributes that can explain the original text. Some can argue that the following $\text{sentence\_attribute}_2$ can also be a valid set of attribute that explains the text:

$$
\text{sentence\_attribute}_2 =  \{\text{classic},\; \text{1990-2000},\; \text{adults},\; \text{france}\}
$$

With this example, we can see that, there are many possible _latent vairables_ that can explain our data with variying levels of likeliness\footnote{For the sake of this example, we'll assume _The Little Prince_ is more likely to be a children's book rather than for adults, so we can say $\text{sentence\_attribute}_1$ is more likely to explain $x$ than, say $\text{sentence\_attribute}_2$}

There are so many possible _latent variables_ that can explain $x$, we have to consider all of them, even the unlikely ones like 

$$
\text{sentence\_attribute}_3 =  \{\text{horror},\; \text{1990-2000},\; \text{babies},\; \text{france}\}
$$

to get the true $P(x)$ by summing over all of them. Because however unlikely, sometimes they are still not technically _impossible_, with a really small probability value. In this example, $\text{sentence\_attribute}$ is constrained into only 4 attributes (genre, year, audience, country) and probably only several labels per category. But in practice, we can deal with even larger dimensions with even more labels per categories or even deal with non-discrete latent variables! When that happens, summing over every possible latent variables

$$
P^\theta(x) = \sum_{z \in \mathcal{Z}} p^\theta(x \mid z)\,p(z)
$$

or in the continuous case,

$$
P^\theta(x) = \int p^\theta(x \mid z)\, p(z)\, dz
$$

becomes very impractical. In the case of **diffusion language models**, the latent variables are often the partially clean sequence, which is a sequence of the same length as the clean inputs, with $|\mathcal{V}|$ possible candidates per position. That's $\text{sequence\_length} \times |\mathcal{V}|$ possible latent variable candidates! This is definitely a problem. Because when we train diffusion language models, we want to compute

$$
-\log(P^\theta(x))
$$

Therefore, we have to find a workaround for this.

## Workaround: Variational Posterior

Let us introduce a _variational posterior_ $q(\text{sentence\_attribute} \mid x)$, which is the distribution of latent variables given the text. For the sake of generality, let's say the _latent vairable_ $\text{(sentence\_attribute)}$ is just some $z$. Note that $\text{sentence\_attribute}$ is just an illustration. In practice, most latent variables are less obvious (it can be a vector of ``meaningless"\footnote{not readily interpretable at face value} numbers, etc.) Of course $q(z \mid x)$ would assign $z$'s that are likelier explanations of $x$'s higher probabilities, like $\text{sentence\_attribute}_1$ for example.

## Derivations

The intractable problem we want to compute is

$$
-\log(P^\theta(x)) = -\log\left(\sum_{z \in \mathcal{Z}} p^\theta(x \mid z)\,p(z)\right)
$$

which we cannot do reasonably, since $\mathcal{Z}$ is too large. We introduce $q(z \mid x)$ into the expression

$$
\begin{aligned}
-\log(P^\theta(x))
&= -\log\left(\sum_{z \in \mathcal{Z}}\frac{q(z \mid x)}{q(z \mid x)} p^\theta(x \mid z)\,p(z)\right) \\
&= -\log\left(\sum_{z \in \mathcal{Z}}q(z \mid x) \frac{p^\theta(x \mid z)\,p(z)}{q(z \mid x)}\right) \\
&= - \log \left(\mathbb{E}_{q(z \mid x)}\left[\frac{p^\theta(x \mid z)\,p(z)}{q(z \mid x)}\right]\right)
\end{aligned} 
$$

In case you (mostly I myself, lol) forgot, how it became an expectation is from

$$
\mathbb{E}_{p(z)}[f(z)] = \int p(z)\, f(z)\, dz
$$

And since log functions are concave, by Jensen's Inequality, we have

$$
\begin{aligned}
    \log \mathbb{E}[\cdot] &\geq \mathbb{E}[\log(\cdot)] \\
    \\
    \log \left(\mathbb{E}_{q(z \mid x)}\left[\frac{p^\theta(x \mid z)\,p(z)}{q(z \mid x)}\right]\right)
    &\geq \mathbb{E}_{q(z \mid x)}\left[\log\left( \frac{p^\theta(x \mid z)\,p(z)}{q(z \mid x)}\right)\right] \\
    \\
    -\log \left(\mathbb{E}_{q(z \mid x)}\left[\frac{p^\theta(x \mid z)\,p(z)}{q(z \mid x)}\right]\right)
    &\leq - \mathbb{E}_{q(z \mid x)}\left[\log\left( \frac{p^\theta(x \mid z)\,p(z)}{q(z \mid x)}\right)\right] \\
\end{aligned}
$$

_That right hand side of the inequality is the ELBO!_ (or NELBO when it's negative). If we minimize that negative ELBO instead, automatically, the actual $-\log P(x)$ would be lower.

## Why Is It Now Tractable?

The key now lies in the fact that the negative ELBO is an **expectation**. When it is an expectation, we can approximate with the Monte Carlo method. Which is basically just sampling several $q(z \mid x)$, computing the log term and then averaging them.
