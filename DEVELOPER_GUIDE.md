# Developer Guide

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    ClusterBasedIndex                     │
│  - buildIndex()                                          │
│  - search()                                              │
└────────────┬────────────────────────────┬────────────────┘
             │                            │
    ┌────────▼────────┐           ┌───────▼────────┐
    │     KMeans      │           │  ClusterIndex  │
    │  - fit()        │           │  - centroids   │
    │  - getCentroids()│          │  - postings    │
    └────────┬────────┘           └───────┬────────┘
             │                            │
    ┌────────▼────────────────────────────▼─────────┐
    │           VectorStorage (Abstract)            │
    │  - addVector()                                │
    │  - getVector()                                │
    │  - getMemorySegment()                         │
    └────────┬──────────────────────┬───────────────┘
             │                      │
    ┌────────▼────────┐     ┌───────▼────────────┐
    │ OnHeapStorage   │     │ OffHeapStorage     │
    │ (HashMap)       │     │ (ByteBuffer)       │
    └─────────────────┘     └────────────────────┘
```

## Module Breakdown

### 1. Clustering (`org.navneev.clustering`)

**KMeans.java**
- Implements Lloyd's k-means algorithm
- Supports both array and VectorStorage input
- Configurable iterations and cluster count

**Key Methods:**
```java
public int[] fit(double[][] data)
public int[] fit(IntegerList vectorIds, VectorStorage storage)
public double[][] getCentroids()
```

### 2. Index (`org.navneev.index`)

**ClusterBasedIndex.java**
- Main entry point for index operations
- Orchestrates clustering and search

**Configuration Constants:**
```java
K_MEANS_ITERATIONS = 300           // Clustering iterations
SAMPLE_SIZE_PCT = 0.1f             // 10% sampling
PCT_OF_CLUSTERS_TO_SEARCH = 0.01f  // 1% cluster search
```

### 3. Storage (`org.navneev.storage`)

**VectorStorage (Abstract)**
- Template method pattern for storage implementations
- Bounds checking in base class
- Subclasses implement actual storage logic

**OnHeapVectorStorage**
- Uses `HashMap<Integer, float[]>`
- Simple, automatic memory management
- Higher GC pressure

**OffHeapVectorsStorage**
- Uses `ByteBuffer.allocateDirect()`
- Lower GC pressure
- Better for large datasets (>100K vectors)

**StorageFactory**
- Factory pattern for storage creation
- Reads `vector.storage` system property

### 4. Model (`org.navneev.model`)

**ClusterIndex**
- Immutable data structure
- Contains centroids, posting lists, and vectors

**IntegerList**
- Dynamic array for integers
- Grows by 2x when capacity exceeded
- Used for posting lists

### 5. Utils (`org.navneev.utils`)

**VectorDistanceCalculationUtils**
- SIMD-optimized distance calculations
- Uses Java Vector API (JDK 23+)
- Multiple overloads for different input types

**EnvironmentUtils**
- Reads system properties
- Debug mode configuration

### 6. Dataset (`org.navneev.dataset`)

**HDF5Reader**
- Reads HDF5 benchmark datasets
- Supports float/double conversion
- Ground truth loading for evaluation

### 7. I/O (`org.navneev.io`)

**ClusterIndexIo**
- Serializes and deserializes ClusterIndex to/from disk
- Uses FileChannel for efficient I/O
- Zero-copy transfers using MemorySegment
- Splits data into two files:
  - `.clus` - Cluster metadata (centroids, posting lists)
  - `.vec` - Vector data (off-heap storage)

**Key Methods:**
```java
public void writeIndex(String fileName, ClusterIndex index)
public ClusterIndex readIndex(String fileName)
```

## Development Setup

### Prerequisites

```bash
# Java 23
java --version

# Gradle 8.x
gradle --version
```

### Build

```bash
# Clean build
./gradlew clean build

# Run tests
./gradlew test

# Run with SIMD
./gradlew run --add-modules jdk.incubator.vector
```

### IDE Setup

**IntelliJ IDEA:**
1. Import as Gradle project
2. Set Project SDK to Java 23
3. Enable preview features
4. Add VM options: `--add-modules jdk.incubator.vector`

**VS Code:**
1. Install Java Extension Pack
2. Configure `settings.json`:
```json
{
  "java.configuration.runtimes": [
    {
      "name": "JavaSE-23",
      "path": "/path/to/jdk-23"
    }
  ]
}
```

## Adding New Features

### Index Serialization

The library supports saving and loading indexes:

```java
// Save index
ClusterBasedIndex index = new ClusterBasedIndex();
index.buildIndex(vectors);
index.serializeIndex("myindex");

// Load index
ClusterBasedIndex loadedIndex = new ClusterBasedIndex();
loadedIndex.deSerializeIndex("myindex");
```

**File Format:**
- Binary format using FileChannel
- Native byte order for optimal performance
- Zero-copy I/O from off-heap memory
- Separate files for metadata and vectors

### Adding a New Distance Metric

1. Add method to `VectorDistanceCalculationUtils`:
```java
public static float manhattanDistance(float[] a, float[] b) {
    // Implementation with SIMD
}
```

2. Update `ClusterBasedIndex` to use new metric:
```java
float dist = VectorDistanceCalculationUtils.manhattanDistance(v1, v2);
```

### Adding a New Storage Type

1. Extend `VectorStorage`:
```java
public class MemoryMappedStorage extends VectorStorage {
    @Override
    protected void addVectorImpl(int id, float[] vector) {
        // Implementation
    }
    
    @Override
    protected float[] getVectorImpl(int id, float[] vector) {
        // Implementation
    }
}
```

2. Update `StorageType` enum:
```java
public enum StorageType {
    ON_HEAP, OFF_HEAP, MEMORY_MAPPED
}
```

3. Update `StorageFactory`:
```java
case MEMORY_MAPPED -> new MemoryMappedStorage(dimensions, totalNumberOfVectors);
```

### Adding a New Clustering Algorithm

1. Create new class in `clustering` package:
```java
public class HierarchicalClustering {
    public int[] fit(VectorStorage storage) {
        // Implementation
    }
}
```

2. Update `ClusterBasedIndex` to support algorithm selection

## Testing

### Unit Tests

```bash
./gradlew test
```

### Benchmark Tests

```bash
# Download SIFT dataset
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5

# Build index
./gradlew run --args="sift-128-euclidean.hdf5" -Dvector.build=true

# Load and search
./gradlew run --args="sift-128-euclidean.hdf5"
```

### Code Formatting

```bash
# Check formatting
./gradlew spotlessCheck

# Apply formatting (Google Java Format - AOSP style)
./gradlew spotlessApply
```

### Performance Profiling

```bash
# With JFR
java -XX:StartFlightRecording=filename=recording.jfr \
     --add-modules jdk.incubator.vector \
     -jar build/libs/cluster-based-ann.jar

# With async-profiler
./profiler.sh -d 30 -f flamegraph.html <pid>
```


## Performance Guidelines

### Memory Management
- Prefer off-heap storage for large datasets
- Reuse arrays where possible
- Avoid unnecessary object allocation in hot paths

### SIMD Optimization
- Use `VectorDistanceCalculationUtils` for all distance calculations
- Ensure arrays are properly aligned
- Profile with `-XX:+PrintAssembly` to verify SIMD usage

### Concurrency
- Current implementation is single-threaded
- Future: Use virtual threads for parallel cluster search
- Ensure thread-safety when adding concurrency

## Debugging

### Enable Debug Output

```bash
./gradlew run -Dvector.debug=true
```

### Enable Index Building

```bash
./gradlew run -Dvector.build=true
```

### Common Issues

**OutOfMemoryError:**
- Increase heap: `-Xmx8g`
- Use off-heap storage: `-Dvector.storage=OFF_HEAP`

**Slow Startup:**
- Use pre-built index: remove `-Dvector.build=true`
- Index deserialization is ~100x faster than building

**Low Recall:**
- Increase `PCT_OF_CLUSTERS_TO_SEARCH`
- Increase `K_MEANS_ITERATIONS`
- Use larger sample size

## Resources

- [Java Vector API Documentation](https://openjdk.org/jeps/460)
- [HDF5 Java Library](https://github.com/jamesmudd/jhdf)
- [ANN Benchmarks](http://ann-benchmarks.com/)
- [FAISS Paper](https://arxiv.org/abs/1702.08734)
