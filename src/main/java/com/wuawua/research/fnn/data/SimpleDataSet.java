package com.wuawua.research.fnn.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;



public class SimpleDataSet<T> implements DataSet<T> {

    private final List<T> records;
    private final int numFeatures;
    private final Random rnd;

    
    public SimpleDataSet() {
        this(0, new Random(), new ArrayList<T>());
    }
    
    /**
     * Constructor.
     *
     * @param numFeatures number of features
     */
    public SimpleDataSet(int numFeatures) {
        this(numFeatures, new Random(), new ArrayList<T>());
    }
    
    /**
     * Constructor.
     *
     * @param numFeatures number of features
     * @param rnd random number generator
     * @param records list of record that compose the data
     */
    public SimpleDataSet(int numFeatures, Random rnd, List<T> records) {
        this.numFeatures = numFeatures;
        this.rnd = rnd;
        this.records = new ArrayList<T>(records);
    }

    
    
    public List<T> getRecords() {
        return records;
    }
    
    public void add(T x) {
        records.add(x);
    }

    @Override
    public int numRecords() {
        return records.size();
    }

    @Override
    public int numFeatures() {
        return numFeatures;
    }

    @Override
    public void shuffle() {
        Collections.shuffle(records, rnd);
    }

    @Override
    public Stream<? extends T> stream() {
        return records.stream();
    }

}
