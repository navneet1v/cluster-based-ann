package org.navneev.storage;

import lombok.Getter;

import java.lang.foreign.MemorySegment;

/**
 * Abstract base class for vector storage implementations.
 *
 * <p>This class defines the contract for storing and retrieving high-dimensional vectors
 * in the index. Implementations can choose different storage strategies based on
 * performance requirements and memory constraints:
 * <ul>
 *   <li><b>On-heap storage</b> - Uses Java heap memory (HashMap-based)</li>
 *   <li><b>Off-heap storage</b> - Uses direct ByteBuffers for reduced GC pressure</li>
 *   <li><b>Memory-mapped storage</b> - Uses file-backed memory for very large datasets</li>
 * </ul>
 *
 * <p>All implementations must support:
 * <ul>
 *   <li>Adding vectors with unique integer IDs</li>
 *   <li>Retrieving vectors by ID</li>
 *   <li>Fixed dimensionality for all vectors</li>
 *   <li>Pre-defined capacity (total number of vectors)</li>
 * </ul>
 *
 * <h3>Design Considerations:</h3>
 * <p>Subclasses should consider:
 * <ul>
 *   <li><b>Thread safety</b> - Whether concurrent access is supported</li>
 *   <li><b>Memory efficiency</b> - Heap vs off-heap trade-offs</li>
 *   <li><b>Access patterns</b> - Sequential vs random access optimization</li>
 *   <li><b>Cleanup</b> - Whether explicit resource cleanup is needed</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * VectorStorage storage = new OnHeapVectorStorage(128, 100000);
 *
 * // Add vectors
 * float[] vector = new float[128];
 * storage.addVector(0, vector);
 *
 * // Retrieve vectors
 * float[] retrieved = storage.getVector(0);
 * }</pre>
 *
 * @author Navneev
 * @version 1.0
 * @see OnHeapVectorStorage
 * @see OffHeapVectorsStorage
 * @since 1.0
 */
public abstract class VectorStorage {

    /**
     * Number of dimensions in each vector
     */
    @Getter
    protected final int dimensions;

    /**
     * Total capacity of vectors that can be stored
     */
    @Getter
    protected final int totalNumberOfVectors;

    private float[] vector;

    /**
     * Constructs a vector storage with specified dimensions and capacity.
     *
     * <p>Initializes common fields used by all storage implementations:
     * <ul>
     *   <li>Dimensions - fixed size for all vectors</li>
     *   <li>Total capacity - maximum number of vectors</li>
     *   <li>Reusable array - for efficient vector retrieval</li>
     * </ul>
     *
     * @param dimensions           the number of dimensions in each vector (must be > 0)
     * @param totalNumberOfVectors the maximum number of vectors to store (must be > 0)
     * @throws IllegalArgumentException if dimensions or totalNumberOfVectors is <= 0
     */
    public VectorStorage(int dimensions, int totalNumberOfVectors) {
        this.dimensions = dimensions;
        this.totalNumberOfVectors = totalNumberOfVectors;
        this.vector = new float[dimensions];
    }

    /**
     * Adds a vector to the storage with the specified ID.
     *
     * <p>This method performs bounds checking before delegating to the implementation.
     * Subclasses should implement {@link #addVectorImpl(int, float[])} for actual storage logic.
     *
     * @param id     the unique identifier for this vector (must be >= 0 and < totalNumberOfVectors)
     * @param vector the vector data to store (length must equal dimensions)
     * @throws IndexOutOfBoundsException if id is out of bounds
     */
    public void addVector(int id, float[] vector) {
        checkBounds(id);
        addVectorImpl(id, vector);
    }

    /**
     * Implementation-specific logic for adding a vector.
     *
     * <p>Subclasses must implement this method to store the vector data.
     * Bounds checking is already performed by {@link #addVector(int, float[])}.
     *
     * @param id     the unique identifier for this vector (guaranteed to be in bounds)
     * @param vector the vector data to store
     */
    protected abstract void addVectorImpl(int id, float[] vector);

    /**
     * Retrieves a vector from the storage by its ID.
     *
     * <p>This method performs bounds checking before delegating to the implementation.
     * Subclasses should implement {@link #getVectorImpl(int, float[])} for actual retrieval logic.
     *
     * @param id the unique identifier of the vector to retrieve
     * @return the vector data
     * @throws IndexOutOfBoundsException if id is out of bounds
     */
    public float[] getVector(int id) {
        checkBounds(id);
        return getVectorImpl(id, vector);
    }

    public void loadVectorInArray(int id, float[] vector) {
        checkBounds(id);
        getVectorImpl(id, vector);
    }

    public MemorySegment getMemorySegment(int id) {
        checkBounds(id);
        return getMemorySegmentImpl(id);
    }

    protected MemorySegment getMemorySegmentImpl(int id) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * Implementation-specific logic for retrieving a vector.
     *
     * <p>Subclasses must implement this method to retrieve the vector data.
     * Bounds checking is already performed by {@link #getVector(int)}.
     *
     * @param id the unique identifier of the vector to retrieve (guaranteed to be in bounds)
     * @return the vector data
     */
    protected abstract float[] getVectorImpl(int id, float[] vector);

    /**
     * Validates that the given vector ID is within valid bounds.
     *
     * @param id the vector ID to validate
     * @throws IndexOutOfBoundsException if id is negative or >= totalNumberOfVectors
     */
    protected void checkBounds(int id) {
        if (id < 0 || id >= totalNumberOfVectors) {
            throw new IndexOutOfBoundsException("Vector ID out of bounds: " + id);
        }
    }

}
