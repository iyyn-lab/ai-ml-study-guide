Linear Algebra is the math of **vectors** (lists of numbers), **matrices** (grids of numbers), and special values like **eigenvalues** that describe transformations.  
It’s the foundation of how computers understand and manipulate data in AI & ML.

## Key Ideas
---
- **Vector**: A list of numbers. **Example: `[2, 3, 5]`**
- **Matrix**: A table of numbers. 
	**Example:**
	```ts
	const matrix = [
	 [1, 2],
	 [3, 4],
	 [5, 6]
	]
	```
- **Eigenvalues & Eigenvectors**: Show how a transformation stretches or shrinks data.

## Why it matters in AI/ML
---
- Datasets are stored as matrices.
- Images, sounds, or text → represented as vectors/matrices.
- Eigenvalues help find **important patterns** in data (like in PCA).

## Real-World Example
---
**Movie Recommendations**
- Each user’s likes can be a vector (e.g., `[Action=5, Comedy=2, Drama=4]`).
- All users together form a big matrix.
- Linear algebra (with eigenvalues) helps find hidden patterns, like “people who like Action + Comedy often also like Sci-Fi,” which powers recommendation systems like Netflix.

**Shoes Store Example**
- Imagine you run a shoe shop.
- Each customer = a vector with info like `[shoe_size, height, weight]`.
- All customers together = a big matrix (your dataset).
- If you want to predict shoe size quickly, you don’t need all details → eigenvalues/eigenvectors help find the **main pattern** (e.g., height matters most).

**Image Processing**
- A grayscale image is just a matrix of pixel values (0–255).
- To rotate the image → multiply the pixel matrix by a **rotation matrix**.
- To compress the image (PCA) → use **eigenvalues & eigenvectors** to keep only the most important features.

**Healthcare Example**
- Patient data: `[height, weight, blood_pressure]` = vector.
- All patients together → matrix.
- PCA on this matrix can reduce 100 medical features down to the **top 5 most important** for predicting diseases.