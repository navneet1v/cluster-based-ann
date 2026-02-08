# Cluster-Based Approximate Nearest Neighbor (ANN) Search

A Java library implementing cluster-based approximate nearest neighbor search using k-means clustering and SIMD-optimized distance calculations.
This is mainly for understanding the algorithm and what potential optimizations you can do. 

## Features

- **K-Means Clustering**: Partitions vector space for efficient search
- **SIMD Optimization**: Leverages Java Vector API for fast distance calculations
- **Off-Heap Storage**: Reduces GC pressure for large datasets
- **HDF5 Support**: Direct integration with standard benchmark datasets
- **Configurable**: Flexible storage and clustering parameters

## Quick Start

### Prerequisites

- Java 23 or higher
- Gradle 8.x

### Build

```bash
./gradlew build
```

### Run

```bash
# With default dataset
./gradlew run

# With custom dataset
./gradlew run --args="path/to/dataset.hdf5"
```

## Algorithm Overview

The library uses a two-phase approach:

1. **Index Building**: K-means clustering partitions vectors into clusters
2. **Search**: Queries search only the most relevant clusters

See [algo_notes.md](algo_notes.md) for detailed algorithm explanation.

## Configuration

### System Properties

- `vector.storage`: Storage type (`ON_HEAP` or `OFF_HEAP`, default: `OFF_HEAP`)
- `vector.debug`: Enable debug output (`true` or `false`, default: `false`)

### Example

```bash
./gradlew run -Dvector.storage=ON_HEAP -Dvector.debug=true
```

## Performance
All performance benchmarks are done on Apple M3 Pro 36GB RAM
Tested on SIFT-128 dataset (1M vectors, 128 dimensions):
- Index build time: 132 seconds  
- Search time (P50): 7ms
- Search time (P90): 8ms 
- Search time (P99): 11ms 
- Recall@100: 0.98

### Iteration 1

```java
private static final int K_MEANS_ITERATIONS = 300;
private static final float SAMPLE_SIZE_PCT = 0.1f;
private static final float PCT_OF_CLUSTERS_TO_SEARCH = 0.01f;
```
```
Index built in 176221 ms, where addTime: 59 ms, build time: 176162 ms

=== Cluster Index Statistics ===
Number of clusters: 1000
Centroid dimensions: 128
Total vectors: 1000000
Empty clusters: 0
Min cluster size: 134
Max cluster size: 3852
Avg cluster size: 1000
=================================

Testing search...
Recall is : 0.804492

Search Time Percentiles:
P50: 1 ms
P90: 2 ms
P99: 3 ms
P100: 44 ms
```

### Iteration 2

```
Starting the build process for index
Index built in 132864 ms, where addTime: 38 ms, build time: 132826 ms

=== Index Configuration ===
K-Means iterations: 300
Sample size: 10.0%
Samples used: 100000
Number of clusters (k): 1000
Clusters to search: 5.0%
Clusters searched per query: 50
Distance metric: Euclidean (squared)
Random seed: 1234212342
Storage type: OffHeapVectorsStorage
===========================


=== Cluster Index Statistics ===
Number of clusters: 1000
Centroid dimensions: 128
Total vectors: 1000000
Empty clusters: 0
Min cluster size: 137
Max cluster size: 3664
Avg cluster size: 1000
=================================

Testing search...

Recall is : 0.9827

Search Time Percentiles:
P50: 7 ms
P90: 8 ms
P99: 11 ms
P100: 88 ms

```

## Project Structure

```
src/main/java/org/navneev/
├── clustering/      # K-means implementation
├── dataset/         # HDF5 data loading
├── index/           # Main index implementation
├── model/           # Data structures
├── sampler/         # Vector sampling
├── storage/         # Vector storage (on/off-heap)
└── utils/           # Distance calculations, utilities
```

## Development

See [developer_guide.md](developer_guide.md) for contribution guidelines and architecture details.

## License

Apache License
