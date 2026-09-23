# Homework: MLPs, Nonlinearity, and Loss Functions

Starter code: `L07/code/xor-problem.ipynb`, `L07/code/mlp-fromscratch__sigmoid-mse.ipynb`, `L07/code/mlp-pytorch_sigmoid-mse.ipynb`, `L07/code/mlp-pytorch_softmax-crossentr.ipynb`

## How this homework works

Below are 8 questions across 3 parts. **You do not need to answer all 8.** Choose **any 4 questions**, with **at least one question from each of the 3 parts (Preferred but not compulsory)**. Pick the ones you find most interesting — all questions are worth equal credit regardless of which ones you choose.

Once you've worked through your chosen questions, **sign up for a 15-minute meeting slot with me** (see course calendar / office hours) to walk me through what you did and what you found. Come ready to show your code and plots and explain your reasoning out loud — this replaces a written "final answer" requirement, so your explanation in the meeting is part of the grade.

## Part 1: Breaking and Fixing the XOR MLP

1. Using `MLPReLU` from `xor-problem.ipynb`, try the number of hidden units at `{1, 2, 5, 50}`. For each value, note whether the model solves XOR (include the decision region plot). What's the smallest number of hidden units you tried that still works?

2. Replace the ReLU activation with `tanh`, and separately with `sigmoid`, keeping hidden units fixed at the value you liked best from Question 1. Compare the three decision boundaries and convergence curves. Which one converges fastest? Which produces the boundary that looks "cleanest" to you?

3. `MLPLinear` has no activation function between its two layers, and it fails to solve XOR. Work through what happens if you multiply out `linear_out(linear_1(x))` by hand for a simple 1-hidden-unit case (i.e., substitute `linear_1(x) = w1*x + b1` into `linear_out`). What single linear layer does this simplify to? Use that to explain in your own words why stacking linear layers without a nonlinearity can never solve XOR, no matter how many hidden units you add.

## Part 2: From-Scratch Backprop

4. In `mlp-fromscratch__sigmoid-mse.ipynb`, the gradients for each weight matrix are computed by hand. Pick **one** weight matrix (e.g. the output layer weights) and, using the code comments/equations already in the notebook as a guide, explain in your own words what each term in the gradient formula represents (e.g., "this term comes from the derivative of the sigmoid," "this term is the error signal from the output").

5. Rebuild the same small network in plain PyTorch, load in the *same* weight values used in the from-scratch version, and call `.backward()`. Compare PyTorch's computed gradient for your chosen weight matrix to the from-scratch value. Do they match (up to floating-point rounding)? What does that tell you about what `.backward()` is doing internally?

   *Optional challenge (not required for credit): extend the from-scratch notebook to a second hidden layer and derive the extra backprop step yourself.*

## Part 3: MSE vs. Cross-Entropy

6. Using the same architecture and dataset, train one model with sigmoid output + MSE loss (`mlp-pytorch_sigmoid-mse.ipynb`) and one with softmax output + cross-entropy loss (`mlp-pytorch_softmax-crossentr.ipynb`). Keep learning rate, epochs, and random seed the same for both.

7. Plot classification accuracy per epoch for both models on the same axes. Which one reaches high accuracy faster?

8. In your own words (a few sentences is fine), why do you think cross-entropy is generally preferred over MSE for classification? *Hint: think about what happens to the training signal when the sigmoid/softmax output is close to 0 or 1 but wrong.*
