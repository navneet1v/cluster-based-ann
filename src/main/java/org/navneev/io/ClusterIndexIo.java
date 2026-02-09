package org.navneev.io;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.navneev.model.ClusterIndex;
import org.navneev.model.IntegerList;
import org.navneev.storage.OffHeapVectorsStorage;
import org.navneev.storage.VectorStorage;

/**
 * Handles serialization and deserialization of ClusterIndex to/from disk. Uses FileChannel and
 * MemorySegment for efficient I/O with zero-copy transfers from off-heap memory.
 */
public class ClusterIndexIo {

    private static final String VECTOR_FILE_EXTENSION = ".vec";
    private static final String CLUSTER_INFO_FILE_EXTENSION = ".clus";

    /**
     * Writes the cluster index to files using FileChannel.
     *
     * @param fileName path to the output file
     * @param index the cluster index to write
     * @throws RuntimeException if write fails
     */
    public void writeIndex(String fileName, ClusterIndex index) {
        try (FileChannel channel =
                FileChannel.open(
                        Path.of(createClusterInfoFileName(fileName)),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {

            // Write centroids
            writeVectorStorage(channel, index.getCentroidStorage());

            // Write posting lists
            writePostingLists(channel, index.getPostingsListArray());

        } catch (IOException e) {
            throw new RuntimeException("Failed to write index: " + e.getMessage(), e);
        }

        // Write vectors to separate file
        String vectorFileName = createVectorFileName(fileName);
        try (FileChannel channel =
                FileChannel.open(
                        Path.of(vectorFileName),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {
            writeVectorStorage(channel, index.getVectors());
        } catch (IOException e) {
            throw new RuntimeException("Failed to write vectors: " + e.getMessage(), e);
        }
    }

    /**
     * Reads the cluster index from files using FileChannel.
     *
     * @param fileName path to the input file
     * @return the deserialized cluster index
     * @throws RuntimeException if read fails
     */
    public ClusterIndex readIndex(final String fileName) {
        try (FileChannel channel =
                FileChannel.open(
                        Path.of(createClusterInfoFileName(fileName)), StandardOpenOption.READ)) {

            // Read centroids
            VectorStorage centroids = readVectorStorage(channel);

            // Read posting lists
            IntegerList[] postingLists = readPostingLists(channel);

            // Read vectors from separate file
            final String vectorFileName = createVectorFileName(fileName);
            VectorStorage vectors;
            try (FileChannel vecChannel =
                    FileChannel.open(Path.of(vectorFileName), StandardOpenOption.READ)) {
                vectors = readVectorStorage(vecChannel);
            }

            return new ClusterIndex(centroids, postingLists, vectors);

        } catch (IOException e) {
            throw new RuntimeException("Failed to read index: " + e.getMessage(), e);
        }
    }

    private void writeVectorStorage(FileChannel channel, VectorStorage storage) throws IOException {
        // Write metadata
        ByteBuffer metadata = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder());
        metadata.putInt(storage.getDimensions());
        metadata.putInt(storage.getTotalNumberOfVectors());
        metadata.flip();
        channel.write(metadata);

        // Write vectors directly from MemorySegment
        for (int i = 0; i < storage.getTotalNumberOfVectors(); i++) {
            MemorySegment segment = storage.getMemorySegment(i);
            ByteBuffer buffer = segment.asByteBuffer().order(ByteOrder.nativeOrder());
            channel.write(buffer);
        }
    }

    private VectorStorage readVectorStorage(FileChannel channel) throws IOException {
        // Read metadata
        ByteBuffer metadata = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder());
        channel.read(metadata);
        metadata.flip();
        int dimensions = metadata.getInt();
        int totalVectors = metadata.getInt();

        VectorStorage storage = new OffHeapVectorsStorage(dimensions, totalVectors);

        // Read vectors directly into MemorySegment
        int vectorSizeBytes = dimensions * Float.BYTES;
        ByteBuffer buffer = ByteBuffer.allocate(vectorSizeBytes).order(ByteOrder.nativeOrder());
        float[] vector = new float[dimensions];

        for (int i = 0; i < totalVectors; i++) {
            buffer.clear();
            channel.read(buffer);
            buffer.flip();
            buffer.asFloatBuffer().get(vector);
            storage.addVector(i, vector);
        }

        return storage;
    }

    private void writePostingLists(FileChannel channel, IntegerList[] postingLists)
            throws IOException {
        // Write length
        ByteBuffer lengthBuf = ByteBuffer.allocate(4).order(ByteOrder.nativeOrder());
        lengthBuf.putInt(postingLists.length);
        lengthBuf.flip();
        channel.write(lengthBuf);

        // Write each posting list
        for (IntegerList list : postingLists) {
            ByteBuffer sizeBuf = ByteBuffer.allocate(4).order(ByteOrder.nativeOrder());
            if (list == null) {
                sizeBuf.putInt(-1);
                sizeBuf.flip();
                channel.write(sizeBuf);
            } else {
                sizeBuf.putInt(list.size());
                sizeBuf.flip();
                channel.write(sizeBuf);

                // Write integers in batches
                ByteBuffer dataBuf =
                        ByteBuffer.allocate(list.size() * 4).order(ByteOrder.nativeOrder());
                for (int i = 0; i < list.size(); i++) {
                    dataBuf.putInt(list.get(i));
                }
                dataBuf.flip();
                channel.write(dataBuf);
            }
        }
    }

    private IntegerList[] readPostingLists(FileChannel channel) throws IOException {
        // Read length
        ByteBuffer lengthBuf = ByteBuffer.allocate(4).order(ByteOrder.nativeOrder());
        channel.read(lengthBuf);
        lengthBuf.flip();
        int length = lengthBuf.getInt();

        IntegerList[] postingLists = new IntegerList[length];

        for (int i = 0; i < length; i++) {
            ByteBuffer sizeBuf = ByteBuffer.allocate(4).order(ByteOrder.nativeOrder());
            channel.read(sizeBuf);
            sizeBuf.flip();
            int size = sizeBuf.getInt();

            if (size == -1) {
                postingLists[i] = null;
            } else {
                IntegerList list = new IntegerList(size);
                ByteBuffer dataBuf = ByteBuffer.allocate(size * 4).order(ByteOrder.nativeOrder());
                channel.read(dataBuf);
                dataBuf.flip();

                for (int j = 0; j < size; j++) {
                    list.add(dataBuf.getInt());
                }
                postingLists[i] = list;
            }
        }

        return postingLists;
    }

    private String createVectorFileName(String fileName) {
        return fileName + VECTOR_FILE_EXTENSION;
    }

    private String createClusterInfoFileName(String fileName) {
        return fileName + CLUSTER_INFO_FILE_EXTENSION;
    }

    public void deleteFilesIfExist(String fileName) {
        try {
            Files.deleteIfExists(Path.of(createClusterInfoFileName(fileName)));
            Files.deleteIfExists(Path.of(createVectorFileName(fileName)));
        } catch (IOException e) {
            throw new RuntimeException("Failed to delete existing file: " + e.getMessage(), e);
        }
    }

    public boolean validateFileExist(String fileName) {
        return Files.exists(Path.of(createVectorFileName(fileName)))
                && Files.exists(Path.of(createClusterInfoFileName(fileName)));
    }
}
