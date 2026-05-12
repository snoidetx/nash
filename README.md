<div align="center">
  <img src="./assets/icon.png" height="120" alt="icon">
  <h1>Is Data Shapley Not Better than Random in Data Selection? Ask NASH</h1>
  <strong>ICML-26 Spotlight</strong> • [<a href="https://arxiv.org/abs/2605.10684">paper</a>]
</div>
<br>

This repository provides the official code implementation for the paper.

## Abstract

Data selection studies the problem of identifying high-quality subsets of training data. While some existing works have considered selecting the subset of data with top-$m$ Data Shapley or other semivalues as they account for the interaction among every subset of data, other works argue that Data Shapley can sometimes perform ineffectively in practice and select subsets that are *no better than random*. This raises the questions: **(I)** *Are there certain "Shapley-informative" settings where Data Shapley consistently works well?* **(II)** *Can we strategically utilize these settings to select high-quality subsets consistently and efficiently?*
In this paper, we propose a novel data selection framework, **NASH** (Non-linear Aggregation of SHapley-informative components), which **(I)** decomposes the target utility function (e.g., validation accuracy) into simpler, Shapley-informative component functions, and selects data by optimizing an objective that **(II)** aggregates these components non-linearly. We demonstrate that NASH substantially boosts the effectiveness of Shapley/semivalue-based data selection with minimal additional runtime cost.

## Note

We are still working on making the documentation clearer. Stay tuned :)!
