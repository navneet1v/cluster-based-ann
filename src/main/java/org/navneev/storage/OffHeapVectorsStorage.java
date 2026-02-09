package org.navneev.storage;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Off-heap storage for vector data using direct ByteBuffer for memory efficiency.
 *
 * <p>This class provides efficient storage for large numbers of high-dimensional vectors by
 * allocating memory outside the Java heap. Benefits include:
 *
 * <ul>
 *   <li>Reduced garbage collection pressure
 *   <li>Better memory locality for vector operations
 *   <li>Predictable memory usage for large datasets
 *   <li>Faster I/O operations when interfacing with native code
 * </ul>
 *
 * <p>The storage is pre-allocated for a fixed number of vectors with fixed dimensions. Vectors are
 * stored sequentially in a contiguous memory block for cache-efficient access.
 *
 * <h3>Memory Layout:</h3>
 *
 * <pre>
 * [vector0_dim0, vector0_dim1, ..., vector0_dimN,
 *  vector1_dim0, vector1_dim1, ..., vector1_dimN,
 *  ...]
 * </pre>
 *
 * <h3>Usage Example:</h3>
 *
 * <pre>{@code
 * OffHeapVectorsStorage storage = new OffHeapVectorsStorage(128, 1000000);
 *
 * // Add vectors
 * float[] vector = new float[128];
 * storage.addVector(vector);
 *
 * // Retrieve vectors
 * float[] retrieved = storage.getVector(0);
 * }</pre>
 *
 * <p><b>Note:</b> This class does not automatically free off-heap memory. The memory will be
 * released when the ByteBuffer is garbage collected, but explicit cleanup may be needed for
 * long-running applications.
 *
 * @author Navneev
 * @version 1.0
 * @since 1.0
 */
public class OffHeapVectorsStorage extends VectorStorage {

    /** Direct buffer for off-heap storage of vector data */
    private final ByteBuffer byteBuffer;

    /** Float view of the byte buffer for convenient float operations */
    private final FloatBuffer floatBuffer;

    private final MemorySegment memorySegment;

    /**
     * Constructs an off-heap storage for vectors with specified dimensions and capacity.
     *
     * <p>Allocates a direct ByteBuffer of size: totalNumberOfVectors × dimensions × 4 bytes. The
     * memory is allocated outside the Java heap and will not be subject to garbage collection until
     * the buffer itself is no longer referenced.
     *
     * <p>The byte order is set to native order for optimal performance when interfacing with native
     * code or SIMD operations.
     *
     * @param dimensions the number of dimensions in each vector (must be > 0)
     * @param totalNumberOfVectors the maximum number of vectors to store (must be > 0)
     * @throws IllegalArgumentException if dimensions or totalNumberOfVectors is <= 0
     * @throws OutOfMemoryError if insufficient off-heap memory is available
     */
    public OffHeapVectorsStorage(int dimensions, int totalNumberOfVectors) {
        super(dimensions, totalNumberOfVectors);
        int sizeInBytes = totalNumberOfVectors * dimensions * Float.BYTES;
        this.byteBuffer = ByteBuffer.allocateDirect(sizeInBytes);
        this.byteBuffer.order(ByteOrder.nativeOrder()); // Set to native byte order
        this.floatBuffer = byteBuffer.asFloatBuffer();
        this.memorySegment = MemorySegment.ofBuffer(byteBuffer);
    }

    /**
     * Adds a vector to the off-heap storage at the specified ID.
     *
     * <p>The vector is copied into the direct buffer at position: id × dimensions. Bounds checking
     * is performed by the parent class.
     *
     * @param id the vector ID (guaranteed to be in bounds by parent class)
     * @param vector the vector data to store (must have length equal to dimensions)
     */
    protected void addVectorImpl(int id, float[] vector) {
        floatBuffer.put(id * dimensions, vector, 0, dimensions);
    }

    /**
     * Retrieves a vector from the off-heap storage by its ID.
     *
     * <p>The vector is read from the buffer at position: id × dimensions. A new array is allocated
     * for each call to ensure thread safety. Bounds checking is performed by the parent class.
     *
     * @param id the zero-based index of the vector to retrieve (guaranteed to be in bounds)
     * @return array containing the vector data (new array on each call)
     */
    protected float[] getVectorImpl(int id, float[] vector) {
        floatBuffer.get(id * dimensions, vector);
        return vector;
    }

    protected MemorySegment getMemorySegmentImpl(int id) {
        long offset = ((long) id) * dimensions * Float.BYTES;
        return memorySegment.asSlice(offset, (long) dimensions * Float.BYTES);
    }
}
