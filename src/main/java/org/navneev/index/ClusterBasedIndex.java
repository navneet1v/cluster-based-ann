package org.navneev.index;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.navneev.clustering.KMeans;
import org.navneev.model.ClusterIndex;
import org.navneev.model.IntegerList;
import org.navneev.sampler.VectorSampler;
import org.navneev.storage.OffHeapVectorsStorage;
import org.navneev.storage.VectorStorage;
import org.navneev.utils.EnvironmentUtils;
import org.navneev.utils.VectorDistanceCalculationUtils;

import java.lang.foreign.MemorySegment;
import java.util.Comparator;
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * Cluster-based index for approximate nearest neighbor search.
 * Uses k-means clustering to partition vectors and enable efficient similarity search.
 */
public class ClusterBasedIndex {

    private static final int K_MEANS_ITERATIONS = 300;
    private static final float SAMPLE_SIZE_PCT = 0.1f;
    private static final float PCT_OF_CLUSTERS_TO_SEARCH = 0.01f;
    private static final long SEED = 1234212342L;
    private static final VectorSampler SAMPLER = new VectorSampler(SEED);
    private ClusterIndex clusterIndex;


    /**
     * Builds the cluster-based index from the given vectors.
     * @param vectors the vector storage containing all vectors to index
     */
    public void buildIndex(final VectorStorage vectors) {
        // 0. Receive vectors in off heap space already.
        // 2. Sample Vectors Ids for K-Means
        final IntegerList sampledVectorIds = SAMPLER.sample(vectors.getTotalNumberOfVectors(),
                (int)(vectors.getTotalNumberOfVectors() * SAMPLE_SIZE_PCT));
        // 3. Run K-Means to get the centroid.
        KMeans kMeans = new KMeans((int)Math.ceil(Math.sqrt(vectors.getTotalNumberOfVectors())), K_MEANS_ITERATIONS);
        // TODO: use the assignment here which is already done later.
        kMeans.fit(sampledVectorIds, vectors);

        final float[][] centroids = kMeans.getCentroids();

        if(EnvironmentUtils.isDebug()) {
            // Compute distances between centroids
            printCentroidDistances(centroids);
        }

        int totalNumberOfCentroids = centroids.length;
        final VectorStorage centroidsStorage = new OffHeapVectorsStorage(centroids[0].length, centroids.length);
        // moving all centroids to off heap, not sure if this is right thing to do here.
        for(int i = 0 ; i < centroidsStorage.getTotalNumberOfVectors(); i++) {
            centroidsStorage.addVector(i, centroids[i]);
        }

        IntegerList[] postingsList = new IntegerList[centroidsStorage.getTotalNumberOfVectors()];

        // 4. Assign Vector Ids to Centroids
        MemorySegment vectorMemorySegment;
        for(int i = 0 ; i < vectors.getTotalNumberOfVectors(); i ++) {
            int assignedCentroid = 0;
            vectorMemorySegment = vectors.getMemorySegment(i);
            float minDistance = VectorDistanceCalculationUtils.euclideanDistance(vectorMemorySegment,
                    centroidsStorage.getMemorySegment(assignedCentroid), vectors.getDimensions());
            for(int j = 1 ; j < totalNumberOfCentroids; j++) {
                float newCentroidDis = VectorDistanceCalculationUtils.euclideanDistance(vectorMemorySegment,
                    centroidsStorage.getMemorySegment(j), vectors.getDimensions());
                // We should see how we want to handle the == case separately. But currently just use this.
                if(newCentroidDis <= minDistance) {
                    minDistance = newCentroidDis;
                    assignedCentroid = j;
                }
            }
            if(postingsList[assignedCentroid] == null) {
                postingsList[assignedCentroid] = new IntegerList();
            }
            postingsList[assignedCentroid].add(i);
        }
        clusterIndex = new ClusterIndex(centroidsStorage, postingsList, vectors);

    }

    /**
     * Prints statistics about the cluster index.
     */
    public void printStats() {
        if(clusterIndex == null) {
            System.out.println("============== Cluster Index is null. Returning ==============");
            return;
        }
        clusterIndex.printClusterIndexStats();
    }

    /**
     * Searches for the top-K nearest neighbors to the query vector.
     * @param queryVector the query vector
     * @param topK number of nearest neighbors to return
     * @return array of vector IDs representing the top-K nearest neighbors
     */
    public int[] search(float[] queryVector, int topK) {
        int minClustersToSearch = Math.max(1, (int)(PCT_OF_CLUSTERS_TO_SEARCH * clusterIndex.getTotalCentroids()));
        // 1. Find minClustersToSearch aka centroids closer to query vectors
        final PriorityQueue<IdAndDistance> closestCentroids = findNearestCentroids(queryVector, minClustersToSearch);

        // 2. Now for each centroid do a brute force search in parallel, TODO: use Virtual threads here
        final PriorityQueue<IdAndDistance> topKQueue = new PriorityQueue<>(topK,
                Comparator.comparingDouble(IdAndDistance::getDistance).reversed());
        while(!closestCentroids.isEmpty()) {
            int cId = closestCentroids.peek().id;
            IntegerList postingList = clusterIndex.getPostingsListArray()[cId];
            if(postingList != null) {
                computeTopKForCentroid(queryVector, topK, postingList, topKQueue);
            }
        }
        // TODO: Merge the results and return topK from different threads
        final int[] finalResults = new int[topKQueue.size()];
        for(int i = topKQueue.size() - 1; i >=0 ; i --) {
            finalResults[i] = Objects.requireNonNull(topKQueue.poll()).getId();
        }
        return finalResults;
    }

    private void computeTopKForCentroid(float[] queryVector, int topK, IntegerList postingsList,
                                        PriorityQueue<IdAndDistance> resultQueue) {
        final VectorStorage flatVectors = clusterIndex.getVectors();
        for(int i = 0 ; i < postingsList.size(); i++) {
            float dis =
                    VectorDistanceCalculationUtils.euclideanDistance(flatVectors.getMemorySegment(postingsList.get(i)), queryVector);
            addWithSizeConstraints(new IdAndDistance(postingsList.get(i), dis), topK, resultQueue);
        }
    }


    private PriorityQueue<IdAndDistance> findNearestCentroids(float[] queryVector, int size) {
        // create a max heap of closest centroids
        PriorityQueue<IdAndDistance> closestCentroids =
                new PriorityQueue<>(size, Comparator.comparingDouble(IdAndDistance::getDistance).reversed());
        int totalNumberOfCentroids = clusterIndex.getTotalCentroids();
        for(int i = 0 ; i < totalNumberOfCentroids; i++) {
            // We can use bulk SIMD here.
            float dis =
                    VectorDistanceCalculationUtils.euclideanDistance(clusterIndex.getCentroidStorage().getMemorySegment(i), queryVector);
            addWithSizeConstraints(new IdAndDistance(i, dis), size, closestCentroids);
        }
        return closestCentroids;
    }

    private void addWithSizeConstraints(IdAndDistance item, int maxSize, PriorityQueue<IdAndDistance> queue) {
        if(queue.size() >= maxSize) {
            assert queue.peek() != null;
            if (queue.peek().getDistance() > item.getDistance()) {
                queue.poll();
                queue.add(item);
            }
        } else {
            queue.add(item);
        }
    }

    private void printCentroidDistances(float[][] centroids) {
        System.out.println("=== Centroid Distance Matrix ===");
        
        // Collect all distances with their centroid pairs
        java.util.List<CentroidPair> distances = new java.util.ArrayList<>();
        for (int i = 0; i < centroids.length; i++) {
            for (int j = i + 1; j < centroids.length; j++) {
                float distance = VectorDistanceCalculationUtils.euclideanDistance(centroids[i], centroids[j]);
                distances.add(new CentroidPair(i, j, distance));
            }
        }
        
        // Sort by distance (smallest to largest)
        distances.sort(Comparator.comparingDouble(CentroidPair::getDistance));
        
        // Print sorted distances
        for (CentroidPair pair : distances) {
            System.out.printf("Distance between centroid %d and %d: %.4f%n", 
                pair.getId1(), pair.getId2(), pair.getDistance());
        }
        System.out.println("=================================");
    }

    @Getter
    @AllArgsConstructor
    public static class IdAndDistance {
        int id;
        float distance;
    }
    
    @Getter
    @AllArgsConstructor
    private static class CentroidPair {
        int id1;
        int id2;
        float distance;
    }
}
