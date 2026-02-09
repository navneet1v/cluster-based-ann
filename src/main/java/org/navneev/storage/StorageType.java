package org.navneev.storage;

/**
 * Enumeration of available vector storage types.
 *
 * <p>Defines the storage strategies available for vector data in the index:
 *
 * <ul>
 *   <li><b>ON_HEAP</b> - Stores vectors in Java heap memory using HashMap
 *   <li><b>OFF_HEAP</b> - Stores vectors in direct ByteBuffer outside Java heap
 * </ul>
 *
 * <p>Used by {@link StorageFactory} to determine which storage implementation to create based on
 * system properties or configuration.
 *
 * <h3>Usage Example:</h3>
 *
 * <pre>{@code
 * // Via system property: -Dvector.storage=ON_HEAP
 * VectorStorage storage = StorageFactory.createStorage(128, 1000000);
 * }</pre>
 *
 * @author Navneev
 * @version 1.0
 * @since 1.0
 * @see StorageFactory
 * @see VectorStorage
 */
public enum StorageType {

    /** On-heap storage using HashMap (automatic memory management) */
    ON_HEAP,

    /** Off-heap storage using direct ByteBuffer (reduced GC pressure) */
    OFF_HEAP;
}
