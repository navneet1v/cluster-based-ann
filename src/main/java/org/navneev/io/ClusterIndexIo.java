package org.navneev.io;

import org.navneev.model.ClusterIndex;
import org.navneev.model.IntegerList;
import org.navneev.storage.OffHeapVectorsStorage;
import org.navneev.storage.VectorStorage;

import java.io.*;

/**
 * Handles serialization and deserialization of ClusterIndex to/from disk.
 */
public class ClusterIndexIo {

    /**
     * Writes the cluster index to a file.
     * @param fileName path to the output file
     * @param index the cluster index to write
     * @throws RuntimeException if write fails
     */
    public void writeIndex(String fileName, ClusterIndex index) {
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(fileName)))) {
            
            // Write centroids
            writeVectorStorage(dos, index.getCentroidStorage());
            
            // Write posting lists
            writePostingLists(dos, index.getPostingsListArray());
            
            // Write vectors
            writeVectorStorage(dos, index.getVectors());
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to write index: " + e.getMessage(), e);
        }
    }

    /**
     * Reads the cluster index from a file.
     * @param fileName path to the input file
     * @return the deserialized cluster index
     * @throws RuntimeException if read fails
     */
    public ClusterIndex readIndex(String fileName) {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(fileName)))) {
            
            // Read centroids
            VectorStorage centroids = readVectorStorage(dis);
            
            // Read posting lists
            IntegerList[] postingLists = readPostingLists(dis);
            
            // Read vectors
            VectorStorage vectors = readVectorStorage(dis);
            
            return new ClusterIndex(centroids, postingLists, vectors);
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to read index: " + e.getMessage(), e);
        }
    }
    
    private void writeVectorStorage(DataOutputStream dos, VectorStorage storage) throws IOException {
        dos.writeInt(storage.getDimensions());
        dos.writeInt(storage.getTotalNumberOfVectors());
        
        for (int i = 0; i < storage.getTotalNumberOfVectors(); i++) {
            float[] vector = storage.getVector(i);
            for (float v : vector) {
                dos.writeFloat(v);
            }
        }
    }
    
    private VectorStorage readVectorStorage(DataInputStream dis) throws IOException {
        int dimensions = dis.readInt();
        int totalVectors = dis.readInt();
        
        VectorStorage storage = new OffHeapVectorsStorage(dimensions, totalVectors);
        float[] vector = new float[dimensions];
        
        for (int i = 0; i < totalVectors; i++) {
            for (int j = 0; j < dimensions; j++) {
                vector[j] = dis.readFloat();
            }
            storage.addVector(i, vector);
        }
        
        return storage;
    }
    
    private void writePostingLists(DataOutputStream dos, IntegerList[] postingLists) throws IOException {
        dos.writeInt(postingLists.length);
        
        for (IntegerList list : postingLists) {
            if (list == null) {
                dos.writeInt(-1);
            } else {
                dos.writeInt(list.size());
                for (int i = 0; i < list.size(); i++) {
                    dos.writeInt(list.get(i));
                }
            }
        }
    }
    
    private IntegerList[] readPostingLists(DataInputStream dis) throws IOException {
        int length = dis.readInt();
        IntegerList[] postingLists = new IntegerList[length];
        
        for (int i = 0; i < length; i++) {
            int size = dis.readInt();
            if (size == -1) {
                postingLists[i] = null;
            } else {
                IntegerList list = new IntegerList(size);
                for (int j = 0; j < size; j++) {
                    list.add(dis.readInt());
                }
                postingLists[i] = list;
            }
        }
        
        return postingLists;
    }
}
