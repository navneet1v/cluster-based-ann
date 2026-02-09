package org.navneev.model;

/**
 * A dynamic array implementation for integers that automatically grows when capacity is exceeded.
 */
public class IntegerList {

    private static final int INITIAL_CAPACITY = 4;
    private static final int GROWTH_FACTOR = 2;
    private int[] array;
    private int size;

    /** Creates an empty list with default initial capacity of 4. */
    public IntegerList() {
        this(INITIAL_CAPACITY);
    }

    /**
     * Creates an empty list with specified initial capacity.
     *
     * @param capacity initial capacity of the list
     */
    public IntegerList(int capacity) {
        array = new int[capacity];
        size = 0;
    }

    /**
     * Returns the number of elements in the list.
     *
     * @return current size of the list
     */
    public int size() {
        return size;
    }

    /**
     * Returns the element at the specified index.
     *
     * @param index the index of the element to return
     * @return the element at the specified index
     * @throws IllegalArgumentException if index is out of bounds
     */
    public int get(int index) {
        checkBounds(index);
        return array[index];
    }

    /**
     * Appends the specified element to the end of this list.
     *
     * @param element the element to be appended
     */
    public void add(int element) {
        if (size == array.length) {
            grow(array.length * GROWTH_FACTOR);
        }
        array[size] = element;
        size++;
    }

    /**
     * Updates the element at the specified index with a new value.
     *
     * @param index the index of the element to update
     * @param value the new value
     * @throws IllegalArgumentException if index is out of bounds
     */
    public void update(int index, int value) {
        checkBounds(index);
        array[index] = value;
    }

    /**
     * Increases the capacity of the internal array to accommodate more elements.
     *
     * @param newCapacity the new capacity of the array
     */
    private void grow(int newCapacity) {
        int[] newArray = new int[newCapacity];
        System.arraycopy(array, 0, newArray, 0, size);
        array = newArray;
    }

    private void checkBounds(int index) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "Index : "
                            + index
                            + " is less than 0 or it is greater than size of "
                            + "list "
                            + size);
        }
    }
}
