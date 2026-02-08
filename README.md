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

### Iteration 1
Tested on SIFT-128 dataset (1M vectors, 128 dimensions):
- Index build time: 170 seconds  
- Search time (P50): TBA 
- Recall@100: TBA

```
Index built in 169788 ms, where addTime : 56 build time: 169732

=== Cluster Index Statistics ===
Number of clusters: 1000
Centroid dimensions: 128
Total vectors: 1000000
Empty clusters: 0
Min cluster size: 2
Max cluster size: 3776
Avg cluster size: 1000
=================================
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
