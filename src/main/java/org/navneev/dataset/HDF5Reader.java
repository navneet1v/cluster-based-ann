package org.navneev.dataset;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;

import java.io.File;

/**
 * Utility class for reading vector datasets and ground truth data from HDF5 files.
 * 
 * <p>This class provides convenient methods to read numerical datasets commonly used
 * in machine learning and vector similarity search applications. It supports automatic
 * type conversion between double and float arrays, and handles common HDF5 file
 * operations with proper resource management.
 * 
 * <p>The class uses the jHDF library to interact with HDF5 files and automatically
 * handles file closing through try-with-resources statements.
 * 
 * <h3>Supported Data Types:</h3>
 * <ul>
 *   <li>2D float arrays for vector datasets</li>
 *   <li>2D int arrays for ground truth neighbor indices</li>
 *   <li>Automatic conversion from double to float precision</li>
 * </ul>
 * 
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * // Read vector dataset
 * float[][] vectors = HDF5Reader.readVectors("data.h5", "train");
 * 
 * // Read ground truth neighbors
 * int[][] neighbors = HDF5Reader.readGroundTruths("data.h5", "neighbors");
 * 
 * // Print file information
 * HDF5Reader.printDatasetInfo("data.h5");
 * }</pre>
 * 
 * @author Navneev
 * @version 1.0
 * @since 1.0
 */
public class HDF5Reader {
    
    /**
     * Reads a 2D float array dataset from an HDF5 file.
     * 
     * <p>This method reads numerical vector data from the specified dataset within
     * the HDF5 file. It supports both float and double precision datasets, automatically
     * converting double arrays to float arrays when necessary.
     * 
     * @param filePath the path to the HDF5 file
     * @param datasetName the name of the dataset within the HDF5 file
     * @return a 2D float array containing the vector data
     * @throws RuntimeException if the file cannot be read, dataset is not found,
     *                         or data type is unsupported
     * @throws IllegalArgumentException if filePath or datasetName is null
     */
    public static float[][] readVectors(String filePath, String datasetName) {
        try (HdfFile hdfFile = new HdfFile(new File(filePath))) {
            Dataset dataset = hdfFile.getDatasetByPath(datasetName);
            
            if (dataset == null) {
                throw new RuntimeException("Dataset '" + datasetName + "' not found in HDF5 file");
            }
            
            Object data = dataset.getData();
            
            if (data instanceof float[][]) {
                return (float[][]) data;
            } else if (data instanceof double[][]) {
                double[][] doubleData = (double[][]) data;
                float[][] floatData = new float[doubleData.length][];
                for (int i = 0; i < doubleData.length; i++) {
                    floatData[i] = new float[doubleData[i].length];
                    for (int j = 0; j < doubleData[i].length; j++) {
                        floatData[i][j] = (float) doubleData[i][j];
                    }
                }
                return floatData;
            } else {
                throw new RuntimeException("Unsupported data type: " + data.getClass());
            }
            
        } catch (Exception e) {
            throw new RuntimeException("Error reading HDF5 file: " + e.getMessage(), e);
        }
    }

    /**
     * Reads a 2D integer array dataset containing ground truth data from an HDF5 file.
     * 
     * <p>This method is typically used to read ground truth neighbor indices for
     * evaluation of nearest neighbor search algorithms. The dataset should contain
     * integer arrays where each row represents the true nearest neighbors for a query.
     * 
     * @param filePath the path to the HDF5 file
     * @param datasetName the name of the dataset within the HDF5 file (e.g., "neighbors")
     * @return a 2D integer array containing ground truth neighbor indices
     * @throws RuntimeException if the file cannot be read, dataset is not found,
     *                         or data type is not integer array
     * @throws IllegalArgumentException if filePath or datasetName is null
     */
    public static int[][] readGroundTruths(String filePath, String datasetName) {
        try (HdfFile hdfFile = new HdfFile(new File(filePath))) {
            Dataset dataset = hdfFile.getDatasetByPath(datasetName);

            if (dataset == null) {
                throw new RuntimeException("Dataset '" + datasetName + "' not found in HDF5 file");
            }

            Object data = dataset.getData();

            if (data instanceof int[][]) {
                int[][] doubleData = (int[][]) data;
                int[][] floatData = new int[doubleData.length][];
                for (int i = 0; i < doubleData.length; i++) {
                    floatData[i] = new int[doubleData[i].length];
                    for (int j = 0; j < doubleData[i].length; j++) {
                        floatData[i][j] = doubleData[i][j];
                    }
                }
                return floatData;
            } else {
                throw new RuntimeException("Unsupported data type: " + data.getClass());
            }

        } catch (Exception e) {
            throw new RuntimeException("Error reading HDF5 file: " + e.getMessage(), e);
        }
    }

    
    /**
     * Prints information about all datasets contained in an HDF5 file.
     * 
     * <p>This utility method displays the file path and lists all available datasets
     * with their dimensions. This is useful for exploring the structure of an HDF5
     * file before reading specific datasets.
     * 
     * <p>Output format:
     * <pre>
     * HDF5 File: /path/to/file.h5
     * Available datasets:
     *   - train [1000, 128]
     *   - test [100, 128]
     *   - neighbors [100, 10]
     * </pre>
     * 
     * @param filePath the path to the HDF5 file to inspect
     * @throws IllegalArgumentException if filePath is null
     */
    public static void printDatasetInfo(String filePath) {
        try (HdfFile hdfFile = new HdfFile(new File(filePath))) {
            System.out.println("HDF5 File: " + filePath);
            System.out.println("Available datasets:");
            hdfFile.getChildren().forEach((name, node) -> {
                if (node instanceof Dataset) {
                    Dataset dataset = (Dataset) node;
                    System.out.println("  - " + name + " " + java.util.Arrays.toString(dataset.getDimensions()));
                }
            });
        } catch (Exception e) {
            System.err.println("Error reading HDF5 file: " + e.getMessage());
        }
    }
}