package org.navneev.model;

import lombok.Getter;
import org.navneev.storage.VectorStorage;

import java.util.Arrays;

/**
 * Represents the cluster index structure for approximate nearest neighbor search.
 * Contains centroids, posting lists mapping centroids to vectors, and the original vectors.
 */
@Getter
public class ClusterIndex {

    private final VectorStorage centroidStorage;

    private final IntegerList[] postingsListArray;

    private final VectorStorage vectors;

    /**
     * Constructs a cluster index with centroids, posting lists, and vectors.
     * @param centroidStorage storage containing cluster centroids
     * @param postingsListArray array mapping each centroid to its assigned vector IDs
     * @param vectors storage containing all original vectors
     */
    public ClusterIndex(final VectorStorage centroidStorage, final IntegerList[] postingsListArray,
                        final VectorStorage vectors) {
        this.centroidStorage = centroidStorage;
        this.postingsListArray = postingsListArray;
        this.vectors = vectors;
    }

    /**
     * Returns the total number of centroids in the index.
     * @return number of centroids
     */
    public int getTotalCentroids() {
        return centroidStorage.getTotalNumberOfVectors();
    }


    /**
     * Prints detailed statistics about the cluster index including cluster sizes and distribution.
     */
    public void printClusterIndexStats() {
        System.out.println("\n\n=== Cluster Index Statistics ===");
        System.out.println("Number of clusters: " + postingsListArray.length);
        System.out.println("Centroid dimensions: " + centroidStorage.getDimensions());
        
        int totalVectors = 0;
        int minClusterSize = Integer.MAX_VALUE;
        int maxClusterSize = 0;
        int emptyClusters = 0;
        int cNumber = -1;
        int printlineCount = 10;
        for (IntegerList cluster : postingsListArray) {
            cNumber++;
            if(cluster == null) {
                System.out.println("Cluster number: " + cNumber + " is null");
                if(printlineCount > 0 ) {
                    System.out.println(Arrays.toString(centroidStorage.getVector(cNumber)));
                    printlineCount--;
                }
                emptyClusters++;
                continue;
            }
            int size = cluster.size();
            totalVectors += size;
            if (size == 0) {
                emptyClusters++;
            } else {
                minClusterSize = Math.min(minClusterSize, size);
                maxClusterSize = Math.max(maxClusterSize, size);
            }
        }
        
        System.out.println("Total vectors: " + totalVectors);
        System.out.println("Empty clusters: " + emptyClusters);
        if (minClusterSize != Integer.MAX_VALUE) {
            System.out.println("Min cluster size: " + minClusterSize);
            System.out.println("Max cluster size: " + maxClusterSize);
            System.out.println("Avg cluster size: " + (totalVectors / (postingsListArray.length - emptyClusters)));
        }
        System.out.println("=================================");
    }

}
