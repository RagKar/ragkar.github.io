# Flame Fractal in Cuda

wrote by @billio\_r and @legoua\_g

## Abstract

Fractals are really interesting objects that can make beautiful image.
One class of these IFS are [fractal flame](https://en.wikipedia.org/wiki/Fractal_flame)
by Scott Draves in 1992.

Our work is a fractal generator using **CUDA** GPU programming to improve
computation time of these iterative objects. Our results are far from
mature software as [Electric Sheep](https://gold.electricsheep.org/) or
[Apophysis](http://www.apophysis.org/). But we will discuss of our
results comparing our CPU and GPU implementation on approach.

## Introduction

For the class of GPU computing we decide to make beautiful Flame
fractal generator as [Flam3](http://flam3.com). Even if there is
already a **LOT** of people who aleady done it, we found this funny and
interesting.

Todo so we base our work on the [paper](http://www.flam3.com/flame_draves.pdf)
written by Draves and Reckase, available on the flame3 site. This paper
give headline to compute fractal flames.

## Definition

Fractal flames differ from ordinary iterated function systems in three
ways:

1. Nonlinear functions are iterated in addition to affine transforms.
2. Log-density display instead of linear or binary.
3. Color by structure (with recursive path) instead of monochrome or
  by density.

A fractal flame is define by a set of flame functions `Fj`.

![Image](https://ragkar.github.io/flamefunction-set.svg)

And each flame function `Fj` is a sum of weighted variations (a list
of the 13 first variations can found in appendix.) These
variation are non-linear (`1`)

![Image](https://ragkar.github.io/flamefunction-def.svg)

It also contain a color and a probability.

At each iterations, a random flame is chosen from there probability.
This flame function is apply on the current position and return a new
position.

Moreover, the color of the pixel at the new position is update (with
these we can see how flame function contribute to a pixel: `2`) and it's
density is increase by one.

The number of iteration is define in advance. It should be a number big
enough in order that the entropy applied.

Once the loop end, some operations are needed to have a better
rendering. One frequent method is to made a supersampling if the
entropy is not good enough and to and to made an average on cells.

Once its done, the log-density (`2`) is used to improve the color and a
gamma can also be used.

## CPU implementation

For the CPU implementation we used C++ but this was a mistake as we
will explain later. The modelisation was pretty simple.

### Variations

Here we used `std::functional`. A variation is defined as:

```
Point variation(double x, double y);
```

### Flame functions

Flame function are based on a simple class as:

```
FlameFunction:
  std::vector<Variation> variations
  std::vector<double> coefficents
  std::vector<double> weights
  Color color
```

### Fractal flames

Same here a fractal flame is a really simple class:

```
FractalFlame:
  std::vector<FlameFunction> flamefunctions
  std::vector<Color> images
```

Here the probability is missing because in practice there is really few
fractal flame who need this.

### Profiling

**TODO**

## GPU implementation

### Where to improve

**TODO**


### Problems

As it was our first time with CUDA we get some troubles with it.

#### Functor

First probleme we got was the use of functor. The `std::function` as
they are function pointer, the function should be `__device__` and
`__host__` but as they are not at the same addresse, we are force to
use indice as it's useless to pass a pointer on a `__host__` function.

Here we use a simple trick. Variations are only define on
`__device__` and an array of functor is define with them. Then when the
kernel is called, it only need an array of index to get the variations
needed.

#### Object-oriented programming

As our first implementation was in C++, our CUDA implementation was
start in object-oriented C++. This was a bad idea as the functor.

We got various trouble with our implementation to move it to CUDA. Then
at one point, we decide to give up on our OO. It was a lot more easier
to use raw array and of double or raw structure.

This come with an insane code reduction and we end with a single kernel
and a single file of 200 lines.

### Profiling

**TODO**
