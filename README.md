# Introduction and motivation

This repository introduces a new type of layer for neural networks:
the _reflection layer_.

The input, output, and kernel/weights of a reflection layer are all of the
same shape and size. This is convenient -- it allows to think of all three
as the same kind of thing (e.g., a vector or an image) and it also makes it
easier to insert a reflection layer into an existing NN architecture.

For example, say we have an RBG image _x_ of shape (256, 256, 3) as input.
Then the weigths _w_ are are of the same shape. Flatten both and call them
_x<sub>f</sub>_ and _w<sub>f</sub>_. Now reflect _x<sub>f</sub>_ across the
orthogonal complement of _w<sub>f</sub>_ and reshape the outcome back to
(256, 256, 3). That's the output. Actually, the reflection used here includes
an overall factor of -1, but that's not crucial -- I am only mentioning this for
those who want to look deeper into the code or at the formulas (which can be
found in the last section). Since the inputs _x_ are images, we can think of the
weigths _w_ as an image as well. Now, what if _w_ itself is plugged in into
the reflection layer as an input? A crucial property of the reflection
operator is that _w_ gets mapped to _w_, i.e. it is left unchanged. However,
an input _x_ that is orthogonal to _w_ will get mapped to _-x_, i.e. it will
be "turned around". Let's repeat this in an imprecise but more
intuitive way: An input _x_ that is very different from _w_ will get mapped
to _-x_.

That last property was my motivation for using reflection operators in neural
networks. Perhaps the learnable weights can pick up the texture of the object
of interest, and then reflection across its orthogonal complement will in some
sense highlight it. However, that's just very vague intuition and things may
not exactly work that way. In fact, I've made some observations that don't
agree with that intuition. Well, I'll try to invesitgate this further. I'd
appreciate any feedback -- so, if you get a chance to throw in the reflection
layer in your NN architecture, I'd be keen to hear how it went :smile:


# Setup


## Virtual environment

Use the conda virtual environment file <code>environment.yml</code>.


## Use of the reflection layer

The notebook <code>reflection_algorithm_demo</code> goes through the steps used
in the reflection layer. It gives some additional explanations (cf. also the 
last section below) on the mathematical steps and also helps read through the
code (there is a lot of broadcasting :flushed:). 


# Results

I once applied the reflection layer to a Kaggle animal classification task,
but I can't post the code as it was written during a gap week at work. The 
reflection layer was inserted into a very shallow NN and this somehow improved
performance (especially when using several reflection layers in parallel and
then combining them). Well, perhaps I'll redo something in this direction at
some point, but for now I am planning to try it on another type of task.

**TO-DO**


# Mathematical background

The operator used in the reflection layer is

<img src="https://render.githubusercontent.com/render/math?math=R_w = 2ww^\top - I,">

which is a reflection across the orthogonal complement of _w_ followed by an
overall multiplication by a factor of -1. Here, _w_ needs to have unit length,
<img src="https://render.githubusercontent.com/render/math?math=||w||=1">,
and _I_ is the identity operator. That is, for an input _x_ we have

<img src="https://render.githubusercontent.com/render/math?math=R_w x = 2w (w^\top x) - x.">

The expression in parentheses is the scalar product of _x_ and _w_ and the
size of that product determines the action of the operator.

Here is my motivation for the reflection layer again, but really, 
inside a complex NN, who knows how things actually play out:
If _w_ is not related to _x_ (say, it's a random vector of length 1),
then the scalar product will be very close to 0 in a high-dimensional
space (e.g. N = 256x256x3), and _x_ gets mapped to something close to
_-x_ (that's the "quite different" or "orthogonal" case mentioned earlier).
If during learning the weights _w_ adopt values that somehow tend to have large
scalar products with one class of inputs _x_ (roughly: weights that somehow
resonate with parts or patterns of one of the classes) then the scalar
product will be larger for that class and the action of the reflection may help
to distinguish it from the other classes.
