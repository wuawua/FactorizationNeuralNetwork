
package com.wuawua.research.nn.data;

import java.util.stream.Stream;

/***
 * DataSet
 * @author Huang Haiping
 *
 * @param <T>
 */
public interface DataSet<T> {
    
    /**
     * @return number of record.
     */
    public int numRecords();
    
    /**
     * Returns Number of features of the record.
     *
     * @return number of features
     */
    public int numFeatures();
    
    /**
     * Shuffle the data, so that stream() results the record in different order.
     */
    public void shuffle();
    
    /**
     * Returns a stream of all record.
     *
     * @return stream of all record
     */
    public Stream<? extends T> stream();
    
}
