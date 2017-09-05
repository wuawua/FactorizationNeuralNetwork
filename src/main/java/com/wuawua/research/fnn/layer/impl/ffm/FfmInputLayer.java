package com.wuawua.research.fnn.layer.impl.ffm;

import java.util.Random;

import com.wuawua.research.fnn.data.DataRecord;
import com.wuawua.research.fnn.data.Feature;
import com.wuawua.research.fnn.layer.Layer;
import com.wuawua.research.fnn.math.Matrix;
import com.wuawua.research.fnn.math.Vector;

public class FfmInputLayer extends Layer<Feature> {
	
    /**
     * Constructor.
     *
     */
    public FfmInputLayer(int dim, float learnRate, Vector bias, Vector regBias, Matrix weights, Vector regWeights) {
        super(dim, learnRate, bias, regBias, weights, regWeights);
    }

    public FfmInputLayer(int dim, int numFeatures, float learnRate, Vector regBias, Vector regWeights, Random rnd, float sdev) {
    	super(dim, numFeatures, learnRate, regBias, regWeights, rnd, sdev);
    }

    @Override
    public Vector forward(DataRecord<Feature> x, Vector hidden) {
    	Vector nextHidden = new Vector(dim);
        for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	float xi = feature.getValue();
        	
        	for(int jj = 0; jj < dim; jj++) {
        		float value =  ((float)xi) * weights.get(ii, jj);
        		nextHidden.add(jj, value);
        	}
        }
        
        
        int size = x.getFeatureSize();
        nextHidden.mul((float)(1/(float)size));
        return nextHidden;
    }

    @Override
    public Vector backward(DataRecord<Feature> x, Vector output, Vector hidden, Vector gradient, double precent) {
    	for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	//float xi = feature.getValue();
    		for(int jj = 0; jj < dim; jj++) {
    			float updateValue = weights.get(ii, jj) - (learnRate * (gradient.get(jj) + regWeights.get(ii) * weights.get(ii, jj)));
            	//System.out.println(ii + " " + updateValue);
    			weights.set(ii, jj, updateValue);
            }
    	}
    	return null;
    }
}