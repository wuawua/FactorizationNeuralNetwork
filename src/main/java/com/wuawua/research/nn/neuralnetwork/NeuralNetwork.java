package com.wuawua.research.nn.neuralnetwork;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.function.DoubleBinaryOperator;
import java.util.logging.Logger;

import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.DataSet;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Vector;


/***
 * 
 * @author Huang Haiping
 *
 */
public class NeuralNetwork<T> {

	protected static final Logger LOG = Logger.getLogger(NeuralNetwork.class.getName());
    protected int numLabels;
    protected Layer<T> inputLayer;
    protected Layer<T> outputLayer;

    public NeuralNetwork(int numLabels, Layer<T> inputLayer, Layer<T> outputLayer) {
        this.numLabels = numLabels;
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
    }
    
    /**
     * Train neural network with record
     * @param record
     */
    public void learn(DataRecord<T> record) {
    }
    
    public void learn(int numIter, DataSet<DataRecord<T>> train, DataSet<DataRecord<T>> test) {
    }
    
    /**
     * Calculate RMSE error 
     * @param test
     * @return
     */
    public double error(DataSet<DataRecord<T>> test) {
        return Math.sqrt(test.stream().mapToDouble(x -> getRMSE(x)).average().getAsDouble());
    }

    /**
     * Calculate RMSE
     * @param record
     * @return
     */
    public double getRMSE(DataRecord<T> record) {
    	double target = 0.0;
    	if(numLabels == 1) {
    		target = record.getTarget();
    	}
    	else {
    		target = record.getTarget() - 1;
    	}
    	Vector hidden = inputLayer.forwardPropagate(record, null);
    	Vector output = outputLayer.forwardPropagate(record, hidden);
    	int label = -1;
    	double max = Double.MIN_VALUE;
    	for(int i = 0; i < numLabels; i++) {
    		if(output.get(i) > max) {
    			label = i;
    			max = predict(output.get(i));
    		}
    	}
    	
    	double predict = 0.0;
    	if(numLabels == 1) {
    		predict = max;
    	}
    	else {
    		predict = (double)label;
    	}
    	
    	DoubleBinaryOperator error = (y, x) -> (y - x) * (y - x);
    	return error.applyAsDouble(predict, target);
    }  
    
    public float predict(float x) {
    	float min = 1.0f;
    	float max = 5.0f;
        return x; //min(max, max(min, x));
    }
}
