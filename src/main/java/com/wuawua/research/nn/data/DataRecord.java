
package com.wuawua.research.nn.data;

import java.util.ArrayList;
import java.util.List;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

/***
 * Data record
 * Record format: target id:field:value
 * example: 5 1:1:0.5 2:2:1.5 3:2:1
 *  
 * @author Huang Haiping
 *
 */
public class DataRecord<T> {

    private float target;
    private List<T> features;
    
    public DataRecord() {
        this.features = new ArrayList<T>();
    }
    
    /**
     * Constructor.
     *
     * @param target target value
     * @param features sparse feature vector
     */
    public DataRecord(float target) {
        this.features = new ArrayList<T>();
        this.target = target;
    }

    /**
     * Get target.
     *
     * @return target
     */
    public float getTarget() {
        return target;
    }
    
    public void setTarget(float target) {
        this.target = target;
    }
    
    public List<T> getFeatures() {
    	return features;
    }
    
    public int getFeatureSize() {
    	return features.size();
    }

    /**
     * Get value of feature
     *
     * @param i feature index
     * @return value of feature
     */
    public T get(int i) {
        return features.get(i);
    }
    
    /**
     * Add new feature
     * @param id
     * @param field
     * @param value
     */
    public void add(T feature) {
        features.add(feature);
    }
    
    public String toString() {
    	StringBuilder content = new StringBuilder();
    	content.append(target).append(" ");
    	for(T feature : features) {
    		content.append( feature.toString()).append(" ");
    	}
    	return content.toString();
    }

}
