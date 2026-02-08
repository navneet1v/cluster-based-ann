# Algorithm Notes

## Overview

This library implements a cluster-based approximate nearest neighbor (ANN) search using k-means clustering as the partitioning strategy. The approach trades perfect accuracy for significant speed improvements.

## Index Building Phase

### 1. Vector Sampling

```
Sample Size = 10% of total vectors
Sampling Method = Reservoir Sampling (uniform distribution)
```

**Why sampling?**
- K-means on full dataset is computationally expensive
- 10% sample provides representative cluster structure
- Reduces clustering time from O(n) to O(0.1n)

### 2. K-Means Clustering

```
Number of Clusters (k) = ⌈√n⌉
Max Iterations = 300
Distance Metric = Euclidean Distance
```

**Algorithm Steps:**

1. **Initialization**: Randomly select k unique vectors as initial centroids
2. **Assignment**: Assign each sampled vector to nearest centroid
3. **Update**: Recalculate centroids as mean of assigned vectors
4. **Convergence**: Repeat until assignments don't change or max iterations reached

**Why √n clusters?**
- Balances cluster granularity vs search cost
- For 1M vectors: ~1000 clusters
- Each cluster contains ~1000 vectors on average

### 3. Vector Assignment

After clustering, assign ALL vectors (not just samples) to their nearest centroid:

```
For each vector v in dataset:
    nearest_centroid = argmin(distance(v, centroid_i))
    posting_list[nearest_centroid].add(v.id)
```

**Data Structure:**
```
ClusterIndex {
    centroids: VectorStorage           // k centroids
    postingsList: IntegerList[]        // k posting lists
    vectors: VectorStorage             // all original vectors
}
```

## Search Phase

### Query Processing

```
Clusters to Search = max(1, 1% of total clusters)
```

**Algorithm Steps:**

1. **Find Nearest Clusters**
   ```
   For each centroid c:
       distance = euclidean_distance(query, c)
       maintain top-k closest centroids in max-heap
   ```

2. **Search Within Clusters**
   ```
   For each selected cluster:
       For each vector_id in posting_list:
           distance = euclidean_distance(query, vector[vector_id])
           maintain top-K results in max-heap
   ```

3. **Return Results**
   ```
   Extract top-K vector IDs from result heap
   ```

## Distance Calculation

### Euclidean Distance (Squared)

```
distance²(a, b) = Σ(aᵢ - bᵢ)²
```

**SIMD Optimization:**

Using Java Vector API (JDK 23+):
```java
// Process 8-16 floats per iteration (depending on CPU)
for (i = 0; i < loopBound; i += vectorLength) {
    va = FloatVector.fromArray(SPECIES, a, i)
    vb = FloatVector.fromArray(SPECIES, b, i)
    diff = va.sub(vb)
    sum += diff.mul(diff).reduceLanes(ADD)
}
```

**Performance Gain:**
- 3-5x faster than scalar implementation
- Utilizes AVX2/AVX-512 instructions when available

## Complexity Analysis

### Index Building

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Sampling | O(n) | O(0.1n) |
| K-Means | O(k × 0.1n × d × iterations) | O(k × d) |
| Assignment | O(n × k × d) | O(n) |
| **Total** | **O(n × k × d)** | **O(n + k × d)** |

Where:
- n = number of vectors
- k = number of clusters (√n)
- d = vector dimensions

### Search

| Operation | Time Complexity |
|-----------|----------------|
| Find Nearest Clusters | O(k × d) |
| Search Clusters | O(0.01k × n/k × d) = O(0.01n × d) |
| **Total** | **O(n × d / 100)** |

**Speedup vs Brute Force:**
- Brute force: O(n × d)
- Cluster-based: O(n × d / 100)
- **~100x faster** (with 1% cluster search)

## Trade-offs

### Accuracy vs Speed

| Clusters Searched | Recall | Speed |
|------------------|--------|-------|
| 1% | ~0.85 | Fast |
| 5% | ~0.95 | Medium |
| 10% | ~0.98 | Slower |
| 100% | 1.00 | Brute Force |

### Memory vs Performance

| Storage Type | Memory | Speed | GC Pressure |
|-------------|--------|-------|-------------|
| On-Heap | Heap | Medium | High |
| Off-Heap | Direct | Fast | Low |

## Optimization Opportunities

### Current Implementation
- Sequential cluster search
- Single-threaded distance calculations
- Fixed cluster search percentage

### Improvements to be added
1. **Parallel Search**: Use virtual threads for concurrent cluster search
2. **Adaptive Search**: Dynamically adjust clusters searched based on query
3. **Graph Refinement**: Build HNSW graph within each cluster

## References

- K-Means Clustering: Lloyd's Algorithm (1957)
- IVF (Inverted File Index): Sivic & Zisserman (2003)
- Java Vector API: JEP 338, 414, 417, 426, 438, 448, 460
