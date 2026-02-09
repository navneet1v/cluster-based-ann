package org.navneev.clustering;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import lombok.Getter;
import org.navneev.model.IntegerList;
import org.navneev.storage.VectorStorage;
import org.navneev.utils.VectorDistanceCalculationUtils;

/**
 * K-means clustering implementation for vector data stored in VectorStorage. Uses Lloyd's algorithm
 * with random initialization.
 */
public class KMeans {
    // Number of clusters
    private final int k;
    // Maximum iterations for convergence
    private final int maxIterations;
    // Cluster centers
    @Getter private float[][] centroids;

    /**
     * Constructs a KMeans clustering instance.
     *
     * @param k number of clusters
     * @param maxIterations maximum number of iterations
     */
    public KMeans(int k, int maxIterations) {
        this.k = k;
        this.maxIterations = maxIterations;
    }

    /**
     * Fits the k-means model to sampled vector data.
     *
     * @param sampledVectorIds list of vector IDs to cluster
     * @param vectors storage containing the vector data
     * @return array of cluster labels for each sampled vector
     */
    public int[] fit(final IntegerList sampledVectorIds, final VectorStorage vectors) {
        int n = sampledVectorIds.size(); // Number of data points
        int[] labels = new int[n]; // Cluster assignment for each point
        centroids = initializeCentroids(sampledVectorIds, vectors); // Initialize k random centroids

        // Iterate until convergence or max iterations
        for (int iter = 0; iter < maxIterations; iter++) {
            int[] newLabels =
                    assignClusters(sampledVectorIds, vectors); // Assign points to nearest centroid
            if (Arrays.equals(labels, newLabels)) break; // Stop if no change in assignments
            labels = newLabels; // Update labels
            updateCentroids(sampledVectorIds, vectors, labels); // Recalculate centroids
        }
        return labels; // Return final cluster assignments
    }

    /**
     * Initializes k centroids by randomly selecting k unique vectors.
     *
     * @param sampleVectorIds list of sampled vector IDs
     * @param vectors storage containing vector data
     * @return array of k initial centroids
     */
    private float[][] initializeCentroids(IntegerList sampleVectorIds, VectorStorage vectors) {
        Random rand = new Random(); // Random number generator
        float[][] cents = new float[k][vectors.getDimensions()]; // Array to store k centroids
        Set<Integer> selected = new HashSet<>(); // Track selected indices
        // Select k unique random points as initial centroids
        for (int i = 0; i < k; i++) {
            int ordinalOfSelectedPoint; // Index of selected point
            do {
                ordinalOfSelectedPoint = rand.nextInt(sampleVectorIds.size()); // Pick random index
            } while (selected.contains(ordinalOfSelectedPoint)); // Ensure uniqueness
            selected.add(ordinalOfSelectedPoint); // Mark as selected

            // Copy data point as centroid
            int vectorDocId = sampleVectorIds.get(ordinalOfSelectedPoint);
            vectors.loadVectorInArray(vectorDocId, cents[i]);
        }
        return cents; // Return initialized centroids
    }

    /**
     * Assigns each sampled vector to the nearest centroid.
     *
     * @param sampleVectorIds list of sampled vector IDs
     * @param vectors storage containing vector data
     * @return array of cluster labels
     */
    private int[] assignClusters(IntegerList sampleVectorIds, VectorStorage vectors) {
        int[] labels = new int[sampleVectorIds.size()]; // Cluster label for each point
        // For each data point
        for (int i = 0; i < sampleVectorIds.size(); i++) {
            float minDist = Float.MAX_VALUE; // Track minimum distance
            // Find nearest centroid
            for (int j = 0; j < k; j++) {
                int vectorDocId = sampleVectorIds.get(i);
                float dist =
                        VectorDistanceCalculationUtils.euclideanDistance(
                                vectors.getMemorySegment(vectorDocId), centroids[j]); // Calculate
                // distance
                if (dist < minDist) { // If closer than current minimum
                    minDist = dist; // Update minimum distance
                    labels[i] = j; // Assign to this cluster
                }
            }
        }
        return labels; // Return cluster assignments
    }

    /**
     * Updates centroids by calculating the mean of all vectors in each cluster.
     *
     * @param sampledVectorIds list of sampled vector IDs
     * @param vectors storage containing vector data
     * @param labels current cluster assignments
     */
    private void updateCentroids(
            IntegerList sampledVectorIds, VectorStorage vectors, int[] labels) {
        int[] counts = new int[k]; // Count points in each cluster
        centroids = new float[k][vectors.getDimensions()]; // Reset centroids
        float[] data = new float[vectors.getDimensions()];
        // Sum all points in each cluster
        for (int i = 0; i < sampledVectorIds.size(); i++) {
            int cluster = labels[i]; // Get cluster assignment
            counts[cluster]++; // Increment count
            // Add point coordinates to cluster sum
            int vectorId = sampledVectorIds.get(i);
            vectors.loadVectorInArray(vectorId, data);
            for (int j = 0; j < data.length; j++) {
                centroids[cluster][j] += data[j];
            }
        }
        // Calculate mean for each cluster
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) { // Avoid division by zero
                // Divide sum by count to get mean
                for (int j = 0; j < centroids[i].length; j++) {
                    centroids[i][j] /= counts[i];
                }
            }
        }
    }
}
