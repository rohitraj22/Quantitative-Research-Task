# Optimal Trade Execution Strategy

### **Project Overview**

This project addresses the challenge of executing a large stock order with minimal cost from market impact. It involves two primary components:
1.  **Modeling Temporary Market Impact:** An empirical model is developed to quantify the relationship between trade size and the resulting price slippage.
2.  **Optimal Execution Algorithm:** A mathematical framework is formulated to determine the optimal trade schedule that minimizes the total execution cost over a given trading day.

The analysis is performed on high-frequency limit order book data for an anonymous ticker, provided in the `FROG.zip` file.

---

### **Methodology**

#### Part 1: Market Impact Modeling

To model the temporary market impact, `g(x)`, we simulate "walking the book" for various trade sizes `x`. The cost is measured as the slippage in basis points (bps) from the mid-price.

Based on empirical analysis and industry standards, we fit a **square-root model** for each one-minute time interval `t`:

$$ g_t(x) \approx \beta_t \sqrt{x} $$

Here, `β_t` is the impact coefficient, which represents the market's liquidity at that time. A higher `β_t` signifies a less liquid market and higher trading costs. These coefficients are estimated via linear regression for each interval.

#### Part 2: Optimal Execution Strategy

The goal is to minimize the total execution cost over `N` periods, subject to buying a total of `S` shares. The optimization problem is:

$$ \min_{x_1, \dots, x_N} \sum_{i=1}^{N} \beta_i x_i^{1.5} \quad \text{subject to} \quad \sum_{i=1}^{N} x_i = S $$

Using the method of Lagrange multipliers, we derive a closed-form solution for the optimal number of shares to trade in each period `i`:

$$ x_i^* = \frac{1}{\beta_i^2} \cdot \frac{S}{\sum_{j=1}^{N} (1/\beta_j^2)} $$

This formula provides the core of our execution algorithm: **trade more aggressively in periods of high liquidity (low `β_i`) and less aggressively in periods of low liquidity (high `β_i`)**.

---

### **Results**

The primary output of the notebook is a series of visualizations that:
1.  Validate the fit of the square-root impact model.
2.  Show the estimated impact parameter `β` evolving throughout the trading day.
3.  Present the final, optimal execution schedule, clearly showing how the trade size is allocated across different periods based on market liquidity.

For a detailed discussion of the methodology and mathematical derivations, please see the PDF documents.
