# README
Code repository for Brochhagen, Franke & van Rooij (2018): Coevolution of lexical meaning and pragmatic use.  Cognitive Science. DOI: [10.1111/cogs.12681](https://doi.org/10.1111%2Fcogs.12681). 

This repository is mantained by Thomas Brochhagen (thomasbrochhagen@gmail.com).

Get in touch if you have any questions!

***

### Requirements

The code is written in python 2.7

A conda environment, named *coevo* and supplied within, is fulfills all the requirements to run the code

To import and activate it, run:

```bash
conda env create --file environment.yaml #import environment from YAML
conda activate coevo #as usual
```



***

### Illustration of dynamics in a reduced type space
The script `2d/2d-plots.py`, illustrates independent and joint effects of the replicator and mutator dynamic using a reduced type space.

Running `coevo/2d/2d-plots.py` generates Figure 2 of the paper. 

The plot is saved in `coevo/plots/2d-dynamics.png`

<p align="center">
  <img width="460" height="300" src="https://raw.githubusercontent.com/brochhagen/coevo/main/plots/2d-dynamics.png">
</p>
. 

The posterior parameters and lambda values that are shown can be changed as arguments of  `quiver_contour()` (l. 240-242)



*** 
 
### Experiments
The files to run the main experiments on full type space are found in `experiments/`

  * Run `run_experiments.py` to automatically compute dynamics for a large number of parameter settings. Otherwise,
  * Call `run_dynamics()` from `rmd.py` to run individual experiments.


 Intermediate computations (U and Q matrices) are saved in `matrices/` to avoid recomputation. 
 
 Both outcomes of individual experimental runs and averages are written to `results/`
 
 **Caution:** Estimating the mutation matrix for the full type space can be computationally expensive. 

***

### Files for type generation

A type is a combination of a lexicon an a linguistic behavior to act on it.

  * `lexica.py` has convenience functions to generate all possible lexica for a number of states and messages; to bin them into categories; and to generate a learning prior over them.
  * `player.py` defines the two player behavior classes we consider: *literal* and *Gricean* 