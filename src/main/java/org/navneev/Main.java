package org.navneev;

import org.navneev.dataset.HDF5Reader;
import org.navneev.index.ClusterBasedIndex;
import org.navneev.storage.StorageFactory;
import org.navneev.storage.VectorStorage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Main entry point for cluster-based ANN index benchmarking and evaluation.
 *
 * <p>This class provides comprehensive testing and performance evaluation of the
 * cluster-based approximate nearest neighbor implementation using HDF5 datasets.
 * The implementation uses k-means clustering to partition the vector space for
 * efficient similarity search. It measures:
 * <ul>
 *   <li>Index construction time (clustering + assignment)</li>
 *   <li>Search performance (percentile analysis)</li>
 *   <li>Recall accuracy against ground truth</li>
 * </ul>
 *
 * <h3>Usage:</h3>
 * <pre>
 * # Run with default dataset
 * ./gradlew run
 *
 * # Run with custom dataset
 * ./gradlew run --args="path/to/dataset.h5"
 * </pre>
 *
 * <h3>Expected HDF5 Dataset Structure:</h3>
 * <pre>
 * dataset.h5
 * ├── train     # Training vectors [N x D]
 * ├── test      # Query vectors [Q x D]
 * └── neighbors # Ground truth neighbors [Q x K]
 * </pre>
 *
 * @author Navneev
 * @version 1.0
 * @since 1.0
 */
public class Main {

    /** Default path to HDF5 dataset file */
    private static String HDF5_FILE_PATH = "sift-128-euclidean.hdf5";

    /**
     * Main entry point for the application.
     *
     * @param args command line arguments; args[0] can specify HDF5 file path
     */
    public static void main(String[] args) {
        if (args.length > 0) {
            HDF5_FILE_PATH = args[0];
        }
        testWithHDF5(HDF5_FILE_PATH);
    }

    /**
     * Tests cluster-based ANN index with HDF5 dataset including construction, search, and evaluation.
     *
     * <p>This method performs a complete benchmark:
     * <ol>
     *   <li>Prints dataset information</li>
     *   <li>Builds cluster-based index using k-means clustering</li>
     *   <li>Executes search queries from test set</li>
     *   <li>Computes recall against ground truth</li>
     *   <li>Reports search time percentiles</li>
     * </ol>
     *
     * @param hdf5FilePath path to the HDF5 dataset file
     */
    private static void testWithHDF5(String hdf5FilePath) {
        System.out.println("Testing Cluster-Based ANN Search with HDF5 file: " + hdf5FilePath);

        HDF5Reader.printDatasetInfo(hdf5FilePath);

        try {
            ClusterBasedIndex index = buildIndex(hdf5FilePath);
            System.gc();

            index.printStats();

            System.out.println("\nTesting search...");
            int k = 100;

            float[][] query_vectors = HDF5Reader.readVectors(hdf5FilePath, "test");

            int[][] actual_results = new int[query_vectors.length][];
            List<Long> searchTimes = new ArrayList<>();
            int counter = 0;
            long startTime;
            for (float[] query : query_vectors) {
                startTime = System.currentTimeMillis();
                int[] topK = index.search(query, k);
                long searchTime = System.currentTimeMillis() - startTime;
                actual_results[counter] = topK;
                searchTimes.add(searchTime);
                counter++;
            }

            float recall = computeRecall(actual_results);
            System.out.println("Recall is : " + recall + "\n");

            printSearchTimePercentiles(searchTimes);
        } catch (Exception e) {
            System.out.println("Error processing HDF5 file: " + e.getMessage());
            e.printStackTrace();
        }
    }


    /**
     * Prints search time percentiles for performance analysis.
     *
     * <p>Calculates and displays:
     * <ul>
     *   <li>P50 (median) - typical search time</li>
     *   <li>P90 - 90th percentile search time</li>
     *   <li>P99 - 99th percentile search time</li>
     *   <li>P100 (max) - worst-case search time</li>
     * </ul>
     *
     * @param searchTimes list of search times in milliseconds
     */
    private static void printSearchTimePercentiles(List<Long> searchTimes) {
        searchTimes.sort(Long::compareTo);
        int size = searchTimes.size();

        long p50 = searchTimes.get((int) (size * 0.50));
        long p90 = searchTimes.get((int) (size * 0.90));
        long p99 = searchTimes.get((int) (size * 0.99));
        long p100 = searchTimes.get(size - 1);

        System.out.println("Search Time Percentiles:");
        System.out.println("P50: " + p50 + " ms");
        System.out.println("P90: " + p90 + " ms");
        System.out.println("P99: " + p99 + " ms");
        System.out.println("P100: " + p100 + " ms");
    }

    /**
     * Computes recall@K metric by comparing search results against ground truth.
     *
     * <p>Recall is calculated as:
     * <pre>
     * recall = (number of correct neighbors found) / (total ground truth neighbors)
     * </pre>
     *
     * <p>A higher recall indicates better search quality. Perfect recall (1.0) means
     * all ground truth neighbors were found in the search results.
     *
     * @param actualResults 2D array of search results [queries][k neighbors]
     * @return recall value between 0.0 and 1.0
     */
    private static float computeRecall(int [][] actualResults) {
        int[][] gt_results = HDF5Reader.readGroundTruths(HDF5_FILE_PATH, "neighbors");
        float neighbors_found = 0.0f;
        for(int i = 0; i < actualResults.length; i++) {
            Set<Integer> gt = Arrays.stream(gt_results[i]).boxed().collect(Collectors.toSet());
            for(int j = 0; j < actualResults[i].length; j++) {
                if(gt.contains(actualResults[i][j])) {
                    neighbors_found++;
                }
            }
        }
        return neighbors_found/(gt_results.length * gt_results[0].length);
    }

    /**
     * Builds cluster-based index from training vectors in the HDF5 dataset.
     *
     * <p>This method:
     * <ol>
     *   <li>Loads training vectors from HDF5 file</li>
     *   <li>Creates vector storage (on-heap or off-heap)</li>
     *   <li>Adds all vectors to storage</li>
     *   <li>Builds cluster-based index using k-means clustering</li>
     *   <li>Reports construction time and progress</li>
     * </ol>
     *
     * <p>The index building process includes:
     * <ul>
     *   <li>Sampling 10% of vectors for k-means</li>
     *   <li>Running k-means with √n clusters</li>
     *   <li>Assigning all vectors to nearest centroids</li>
     *   <li>Creating inverted index (posting lists)</li>
     * </ul>
     *
     * <p>Progress is printed every 100,000 vectors to monitor long-running builds.
     *
     * @param hdf5FilePath path to the HDF5 dataset file
     * @return constructed cluster-based index ready for search
     * @throws RuntimeException if HDF5 file cannot be read
     */
    private static ClusterBasedIndex buildIndex(final String hdf5FilePath) {
        float[][] vectors = HDF5Reader.readVectors(hdf5FilePath, "train");
        System.out.println("\nLoaded " + vectors.length + " vectors with dimension " + vectors[0].length);

        final ClusterBasedIndex clusterBasedIndex = new ClusterBasedIndex();

        VectorStorage vectorStorage = StorageFactory.createStorage(vectors[0].length, vectors.length);
        int numVectors = vectors.length;
        System.out.println("Adding " + numVectors + " vectors to Storage...");
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < numVectors; i++) {
            vectorStorage.addVector(i, vectors[i]);
            if ((i + 1) % 100000 == 0) {
                System.out.println("Added " + (i + 1) + " vectors to storage");
            }
        }
        long addTime = System.currentTimeMillis() - startTime;
        vectors = null;
        System.gc();

        startTime = System.currentTimeMillis();
        System.out.println("Starting the build process for index");
        clusterBasedIndex.buildIndex(vectorStorage);
        long buildTime = System.currentTimeMillis() - startTime;

        System.out.println("Index built in " + (buildTime + addTime) + " ms, where addTime: " + addTime + " ms, build" +
                " time: " + buildTime + " ms");
        return clusterBasedIndex;
    }
}