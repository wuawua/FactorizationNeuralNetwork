package com.wuawua.research.nn.layer.impl.ffm;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Random;

import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;


public class FfmOutputLayer extends Layer<Feature> {
	
    private int numLabels;
    private int numFields;
    private final int SINGLE_LABEL_NUMBER = 1;
    private Matrix fieldWeights;

    /**
     * Constructor.
     *
     * @param b initial bias
     * @param w initial feature weight vector
     * @param m initial feature interaction matrix
     */
    public FfmOutputLayer(int dim, int numLabels, int numFields, float learnRate, Vector bias, Vector regBias, Matrix weights, Vector regWeights) {
    	super(dim, learnRate, bias, regBias, weights, regWeights, null);
    	this.numLabels = numLabels;
    	this.numFields = numFields;
    }

    public FfmOutputLayer(int dim, int numLabels, int numFeatures, int numFields, float learnRate, Vector regBias, Vector regWeights, Random rnd, float sdev) {  
    	super(dim, numFeatures * numLabels * numFields, learnRate, regBias, regWeights, rnd, sdev);
    	this.numLabels = numLabels;
    	this.numFields = numFields;
    }
    
    @Override
    public Vector forwardPropagate(DataRecord<Feature> record, Vector hidden) {
    	return null;
    }
    
    @Override
	public Matrix forwardPropagate(DataRecord<Feature> record, Matrix hidden) {
    	Matrix output = new Matrix(numLabels, 1);
    	output.zero();
    	
    	//Summary all xi's weights group by field
    	fieldWeights = new Matrix(numFields, dim);
    	for(Feature feature : record.getFeatures()) {
    		for(int labelIndex = 0; labelIndex < numLabels; labelIndex++) {    		
    			int ii = (int)feature.getID();
    			float xi = feature.getValue();
    			
    			//Bias
    			output.add(labelIndex, 0, bias.get(ii));
    			
		    	for(int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
		        	int weightRowIndex = ii * numLabels * numFields + labelIndex * numFields +  fieldIndex;
		    		for(int jj = 0; jj < dim; jj++) {
		    			float value = xi * weights.get(weightRowIndex, jj);
		    			fieldWeights.add(fieldIndex, jj, value);
		    		}
		    	}
    		}
    	}
    	
    	//Calculate output 
		for(int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
			for(int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
    			for(int jj = 0; jj < dim; jj++) {
    				output.add(labelIndex, 0, hidden.get(fieldIndex, jj) * fieldWeights.get(fieldIndex, jj));
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
    
    public Matrix computeSoftmax(Matrix output) {		
		double max = output.get(0,0);
		Matrix result = new Matrix(output.getRows(), 0);
		double z = 0.0f;
		for (int i = 0; i < numLabels; i++) {
			max = Math.max(output.get(i, 0), max);
		}
		for (int i = 0; i < numLabels; i++) {
			output.add(i, 0, (float)Math.exp(output.get(i, 0) - max));
			z += output.get(i, 0);
		}
		for (int i = 0; i < numLabels; i++) {
			result.add(i, 0, (float)(output.get(i, 0) / z));
		}
		return result;
	}
        
    @Override
    public Vector backwardPropagate(DataRecord<Feature> x, Vector output, Vector hidden, Vector gradient) {
    	return null;
    }
    
    @Override
	public Matrix backwardPropagate(DataRecord<Feature> record, Matrix output, Matrix hidden, Matrix gradient) {
    	Matrix nextGradient = new Matrix(numFields, dim);
    	int target = (numLabels == SINGLE_LABEL_NUMBER)?(int)(record.getTarget()):(int)(record.getTarget() - 1);

    	for(int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
    		float label = (labelIndex == target) ? 1.0f : 0.0f;
			if(numLabels == 1) {
				label = target;
			}
			else {
				label = (labelIndex == target) ? 1.0f : 0.0f;
			}
			float lambda = predict(output.get(labelIndex, 0)) - label;
			
			//Gradient
			for(int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
	    		for(int jj = 0; jj < dim; jj++) {
	    			nextGradient.add(fieldIndex, jj, lambda * fieldWeights.get(fieldIndex, jj));
	    		}
	    	}
			
			//Update xi's weights
    		for(Feature feature : record.getFeatures()) {
	        	int ii = (int)feature.getID();
	        	float xi = feature.getValue();
	        	
	        	float biasUpdate = -learnRate * (lambda + regBias.get(ii) * bias.get(ii));
    			bias.add(ii, biasUpdate);
    			
	        	for(int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
		    		for(int jj = 0; jj < dim; jj++) {
		    			int weightRowIndex = ii * numLabels * numFields + labelIndex * numFields + fieldIndex;
		    			float updateWeight = -(float)(learnRate * (lambda * hidden.get(fieldIndex, jj) * xi + regWeights.get(ii) * weights.get(weightRowIndex,jj)));
		    			weights.add(weightRowIndex, jj, updateWeight);
		            }
	        	}
    		}
    	}
    	
    	//nextGradient.mul((float)(1/(float)numLabels));
    	return nextGradient;
	}
    
    public float predict(float x) {
    	float min = 1.0f;
    	float max = 5.0f;
        return x; //min(max, max(min, x));
    }


	@Override
	public void accumulateGradient(Vector grad) {
	}

	

	
}
