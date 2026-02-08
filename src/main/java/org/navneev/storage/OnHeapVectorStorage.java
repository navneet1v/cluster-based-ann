package org.navneev.storage;

import java.util.HashMap;
import java.util.Map;

/**
 * On-heap storage implementation for vector data using HashMap.
 * 
 * <p>This class provides vector storage within the Java heap using a HashMap for
 * fast random access by vector ID. Suitable for:
 * <ul>
 *   <li>Small to medium-sized datasets that fit comfortably in heap</li>
 *   <li>Applications where garbage collection overhead is acceptable</li>
 *   <li>Scenarios requiring simple memory management</li>
 *   <li>Development and testing environments</li>
 * </ul>
 * 
 * <p>Each vector is stored as a separate float array in the heap, providing:
 * <ul>
 *   <li>O(1) average-case lookup by ID</li>
 *   <li>Automatic memory management via garbage collection</li>
 *   <li>Simple implementation without native memory concerns</li>
 * </ul>
 * 
 * <h3>Trade-offs vs Off-Heap Storage:</h3>
 * <table border="1">
 *   <tr><th>Aspect</th><th>On-Heap</th><th>Off-Heap</th></tr>
 *   <tr><td>GC Pressure</td><td>Higher</td><td>Lower</td></tr>
 *   <tr><td>Memory Management</td><td>Automatic</td><td>Manual</td></tr>
 *   <tr><td>Cache Locality</td><td>Lower</td><td>Higher</td></tr>
 *   <tr><td>Implementation</td><td>Simple</td><td>Complex</td></tr>
 * </table>
 * 
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * OnHeapVectorStorage storage = new OnHeapVectorStorage(128, 100000);
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
 * @since 1.0
 * @see VectorStorage
 * @see OffHeapVectorsStorage
 */
public class OnHeapVectorStorage extends VectorStorage {

    /** HashMap storing vector ID to vector data mapping */
    private final Map<Integer, float[]> idToVectorMap;

    /**
     * Constructs an on-heap vector storage with specified dimensions and capacity.
     * 
     * <p>Initializes a HashMap with the specified initial capacity to minimize
     * rehashing operations during vector insertion.
     * 
     * @param dimensions the number of dimensions in each vector (must be > 0)
     * @param totalNumberOfVectors the expected number of vectors (used for HashMap sizing)
     * @throws IllegalArgumentException if dimensions or totalNumberOfVectors is <= 0
     */
    public OnHeapVectorStorage(int dimensions, int totalNumberOfVectors) {
        super(dimensions, totalNumberOfVectors);
        idToVectorMap = new HashMap<>(totalNumberOfVectors);
    }

    /**
     * Adds a vector to the storage with the specified ID.
     * 
     * <p>Creates a defensive copy of the input vector to prevent external modifications
     * from affecting the stored data. The copy is stored in the HashMap with the given ID.
     * 
     * <p>If a vector with the same ID already exists, it will be replaced.
     * 
     * @param id the unique identifier for this vector
     * @param vector the vector data to store (will be copied)
     */
    @Override
    public void addVectorImpl(int id, float[] vector) {
        float[] copiedVector = new float[vector.length];
        System.arraycopy(vector, 0, copiedVector, 0, vector.length);
        idToVectorMap.put(id, copiedVector);
    }

    /**
     * Retrieves a vector from the storage by its ID.
     * 
     * <p>Returns a direct reference to the stored vector array. Modifications to the
     * returned array will affect the stored data.
     * 
     * <p><b>Warning:</b> The returned array is not a copy. If you need to modify the
     * vector without affecting the storage, create a copy first.
     * 
     * @param id the unique identifier of the vector to retrieve
     * @return the vector data, or null if no vector exists with the given ID
     */
    @Override
    public float[] getVectorImpl(int id, float[] vector) {
        return idToVectorMap.get(id);
    }
}
