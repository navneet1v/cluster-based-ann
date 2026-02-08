package org.navneev.sampler;

import org.navneev.model.IntegerList;

import java.util.Random;

/**
 * Samples vector IDs using reservoir sampling algorithm.
 * Guarantees uniform distribution without replacement.
 */
public class VectorSampler {
    
    private final Random random;
    
    /**
     * Creates sampler with specified seed for reproducible results.
     * @param seed random seed
     */
    public VectorSampler(long seed) {
        this.random = new Random(seed);
    }
    
    /**
     * Samples vector IDs using reservoir sampling.
     * @param numVectors total number of vectors (0 to numVectors-1)
     * @param sampleSize number of IDs to sample
     * @return array of sampled vector IDs
     */
    public IntegerList sample(int numVectors, int sampleSize) {
        if (sampleSize >= numVectors) {
            return range(numVectors);
        }

        IntegerList reservoir = new IntegerList(sampleSize);
        //1. Initialize reservoir
        for (int i = 0; i < sampleSize; i++) {
            reservoir.add(i);
        }

        // 2. Process other elements now.
        for (int i = sampleSize; i < numVectors; i++) {
            // to generate values from [0 to i]
            int j = random.nextInt(i + 1);
            if(j < sampleSize) {
                reservoir.update(j, i);
            }
        }
        return reservoir;
    }

    private IntegerList range(int n) {
        IntegerList result = new IntegerList(n);
        for (int i = 0; i < n; i++) {
            result.add(i);
        }
        return result;
    }
}