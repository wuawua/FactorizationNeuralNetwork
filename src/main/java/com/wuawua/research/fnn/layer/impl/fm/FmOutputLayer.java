package com.wuawua.research.fnn.layer.impl.fm;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Random;
import com.wuawua.research.fnn.data.DataRecord;
import com.wuawua.research.fnn.data.Feature;
import com.wuawua.research.fnn.layer.Layer;
import com.wuawua.research.fnn.math.Matrix;
import com.wuawua.research.fnn.math.Vector;


public class FmOutputLayer extends Layer<Feature> {
	
    private int numLabels;
    private final int SINGLE_LABEL_NUMBER = 1;
    private Vector featureWeights;
    private float b = 0;

    /**
     * Constructor.
     *
     * @param b initial bias
     * @param w initial feature weight vector
     * @param m initial feature interaction matrix
     */
    public FmOutputLayer(int dim, int numLabels, float learnRate, Vector bias, Vector regBias, Matrix weights, Vector regWeights) {
    	super(dim, learnRate, bias, regBias, weights, regWeights);
    	this.numLabels = numLabels;
    	this.featureWeights = new Vector(weights.getRows());
    }

    public FmOutputLayer(int dim, int numLabels, int numFeatures, float learnRate, Vector regBias, Vector regWeights, Random rnd, float sdev) {  
    	super(dim, numFeatures * numLabels, learnRate, regBias, regWeights, rnd, sdev);
    	this.numLabels = numLabels;
    	this.featureWeights = new Vector(numFeatures);
    }
    
    @Override
    public Vector forward(DataRecord<Feature> x, Vector hidden) {
    	Vector output = new Vector(numLabels);
    	output.zero();
    	for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	float xi = feature.getValue();
    		for(int index = 0; index < numLabels; index++) {
    			output.add(index, bias.get(ii));
    			for(int jj = 0; jj < dim; jj++) {
    				output.add(index, hidden.get(jj) * xi * weights.get(ii * numLabels + index, jj));
    			}
    		}
    	}

    	if(numLabels == SINGLE_LABEL_NUMBER) {
    		return output;
    	}
    	else {
    		return computeSoftmax(output);
    	}
    }
    
    public Vector computeSoftmax(Vector output) {		
		double max = output.get(0);
		Vector result = new Vector(output.getRows());
		double z = 0.0f;
		for (int i = 0; i < numLabels; i++) {
			max = Math.max(output.get(i), max);
		}
		for (int i = 0; i < numLabels; i++) {
			output.add(i, (float)Math.exp(output.get(i) - max));
			z += output.get(i);
		}
		for (int i = 0; i < numLabels; i++) {
			result.add(i, (float)(output.get(i) / z));
		}
		return result;
	}
        
    @Override
    public Vector backward(DataRecord<Feature> x, Vector output, Vector hidden, Vector gradient, double percent) {
    	Vector nextGradient = new Vector(dim);
    	int target = (numLabels == SINGLE_LABEL_NUMBER)?(int)(x.getTarget()):(int)(x.getTarget() - 1);

    	for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	float xi = feature.getValue();
    		for(int index = 0; index < numLabels; index++) {
    			float label = (index == target) ? 1.0f : 0.0f;
    			if(numLabels == 1) {
    				label = target;
    			}
    			else {
    				label = (index == target) ? 1.0f : 0.0f;
    			}
    			float lambda = predict(output.get(index)) - label;
    			
    			float biasUpdate = -learnRate * (lambda + regBias.get(ii) * bias.get(ii));
    			bias.add(ii, biasUpdate);
    			
	    		for(int jj = 0; jj < dim; jj++) {
	    			int wIndex = ii * numLabels + index;
	    			nextGradient.add(jj, lambda * xi * weights.get(wIndex, jj));
	    			float updateWeight = -(float)(learnRate * (lambda * hidden.get(jj) * xi + regWeights.get(ii) * weights.get(wIndex,jj)));
	    			weights.add(wIndex, jj, updateWeight);
	            }
    		}
    	}
    	
    	int size = x.getFeatureSize();
    	nextGradient.mul((float)(1/(float)size));
    	return nextGradient;
    }
    
    public float predict(float x) {
    	float min = 1.0f;
    	float max = 5.0f;
        return x; //min(max, max(min, x));
    }
}
