package org.navneev.storage;

/**
 * Factory class for creating VectorStorage instances based on system configuration.
 *
 * <p>This factory reads the "vector.storage" system property to determine which storage
 * implementation to instantiate. Supports both on-heap and off-heap storage.
 *
 * <h3>Usage Example:</h3>
 *
 * <pre>{@code
 * // Use default (OFF_HEAP)
 * VectorStorage storage = StorageFactory.createStorage(128, 1000000);
 *
 * // Or configure via system property
 * // -Dvector.storage=ON_HEAP
 * VectorStorage storage = StorageFactory.createStorage(128, 1000000);
 * }</pre>
 *
 * @author Navneev
 * @version 1.0
 * @since 1.0
 * @see VectorStorage
 * @see OnHeapVectorStorage
 * @see OffHeapVectorsStorage
 */
public class StorageFactory {

    private static final String VECTOR_STORAGE_KEY = "vector.storage";
    private static final String DEFAULT_VECTOR_STORAGE = "OFF_HEAP";

    /**
     * Creates a VectorStorage instance based on system property configuration.
     *
     * <p>Reads the "vector.storage" system property to determine storage type. Valid values are
     * "ON_HEAP" or "OFF_HEAP". Defaults to OFF_HEAP if not specified.
     *
     * @param dimensions the dimensionality of vectors to store
     * @param totalNumberOfVectors the maximum number of vectors to store
     * @return a VectorStorage instance (OnHeapVectorStorage or OffHeapVectorsStorage)
     * @throws IllegalArgumentException if vector.storage property has invalid value
     */
    public static VectorStorage createStorage(int dimensions, int totalNumberOfVectors) {
        final StorageType storageType =
                StorageType.valueOf(System.getProperty(VECTOR_STORAGE_KEY, DEFAULT_VECTOR_STORAGE));

        return switch (storageType) {
            case StorageType.ON_HEAP -> new OnHeapVectorStorage(dimensions, totalNumberOfVectors);
            case StorageType.OFF_HEAP ->
                    new OffHeapVectorsStorage(dimensions, totalNumberOfVectors);
        };
    }
}
