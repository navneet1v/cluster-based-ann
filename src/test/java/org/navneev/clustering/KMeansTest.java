package org.navneev.clustering;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.navneev.sampler.VectorSampler;
import org.navneev.storage.OffHeapVectorsStorage;
import org.navneev.storage.VectorStorage;

class KMeansTest {

    private static final VectorSampler SAMPLER = new VectorSampler(123L);

    @Test
    void testSimpleClustering() {
        float[][] data = {
            {1.0f, 1.0f},
            {1.5f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 7.0f},
            {3.5f, 5.0f},
            {4.5f, 5.0f},
            {3.5f, 4.5f}
        };

        VectorStorage vectorStorage = new OffHeapVectorsStorage(2, data.length);
        for (int i = 0; i < data.length; i++) {
            vectorStorage.addVector(i, data[i]);
        }

        KMeans kmeans = new KMeans(2, 100);
        int[] labels = kmeans.fit(SAMPLER.sample(data.length, data.length), vectorStorage);

        assertEquals(7, labels.length);
        assertTrue(labels[0] == labels[1]);
        assertTrue(labels[2] == labels[3] || labels[2] == labels[4]);
    }

    @Test
    void testSingleCluster() {
        float[][] data = {{1.0f, 2.0f}, {1.1f, 2.1f}, {0.9f, 1.9f}};
        VectorStorage vectorStorage = new OffHeapVectorsStorage(2, data.length);
        for (int i = 0; i < data.length; i++) {
            vectorStorage.addVector(i, data[i]);
        }

        KMeans kmeans = new KMeans(1, 10);
        int[] labels = kmeans.fit(SAMPLER.sample(data.length, data.length), vectorStorage);

        assertEquals(3, labels.length);
        assertEquals(0, labels[0]);
        assertEquals(0, labels[1]);
        assertEquals(0, labels[2]);
    }

    @Test
    void testCentroidsShape() {
        float[][] data = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};

        VectorStorage vectorStorage = new OffHeapVectorsStorage(3, data.length);
        for (int i = 0; i < data.length; i++) {
            vectorStorage.addVector(i, data[i]);
        }

        KMeans kmeans = new KMeans(2, 10);
        kmeans.fit(SAMPLER.sample(data.length, data.length), vectorStorage);

        float[][] centroids = kmeans.getCentroids();
        assertEquals(2, centroids.length);
        assertEquals(3, centroids[0].length);
    }

    @Test
    void testExactClusters() {
        float[][] data = {
            {0.0f, 0.0f},
            {0.0f, 0.0f},
            {10.0f, 10.0f},
            {10.0f, 10.0f}
        };

        VectorStorage vectorStorage = new OffHeapVectorsStorage(2, data.length);
        for (int i = 0; i < data.length; i++) {
            vectorStorage.addVector(i, data[i]);
        }

        KMeans kmeans = new KMeans(2, 100);
        int[] labels = kmeans.fit(SAMPLER.sample(data.length, data.length), vectorStorage);

        assertEquals(labels[0], labels[1]);
        assertEquals(labels[2], labels[3]);
        assertNotEquals(labels[0], labels[2]);
    }
}
