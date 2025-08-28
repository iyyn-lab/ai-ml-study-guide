Optimization is about finding the **best solution** from many possibilities.  
In ML, it means adjusting model parameters so predictions are as accurate as possible.

## Key Ideas
---
- **Objective / Loss Function** → A formula that measures error (how wrong the model is).
- **Gradient Descent** → Step-by-step method to reduce error by moving in the direction of improvement.
- **Local vs Global Minimum** → Sometimes you get “stuck” in a good-but-not-best solution (local minimum) instead of the overall best (global minimum).

## Why it matters in AI/ML
---
- Every ML model needs optimization to learn.
- Neural networks, linear regression, logistic regression — all use optimization to tune weights.
- Without optimization, the model won’t improve.

## Real-World Example
---
**Cyclist on a Hill**
- Imagine a cyclist trying to reach the lowest point in a valley.
- The slope (gradient) tells the cyclist which way to go downhill.
- Step by step, they move down until they reach the **lowest point**.
- In ML, the “lowest point” = **minimum error** → the best model performance.