package com.wuawua.research.nn.neuralnetwork.impl;

import java.util.function.DoubleBinaryOperator;
import java.util.logging.Logger;

import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.DataSet;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.neuralnetwork.NeuralNetwork;


/***
 * 
 * @author Huang Haiping
 *
 */
public class FfmNeuralNetwork extends NeuralNetwork<Feature> {

    private static final Logger LOG = Logger.getLogger(FfmNeuralNetwork.class.getName());
    private int trainNum = 0;
    
    public FfmNeuralNetwork(int numLabels, Layer<Feature> inputLayer, Layer<Feature> outputLayer) {
    	super(numLabels, inputLayer, outputLayer);
    }
    
    /**
     * Train neural network with record
     * @param record
     */
    @Override
    public void learn(DataRecord<Feature> record) {
    	double percent = 1.0;
    	
    	//Forward
    	Matrix hidden = inputLayer.forwardPropagate(record, new Matrix());
    	Matrix output = outputLayer.forwardPropagate(record, hidden);
    	
    	//Backward
    	Matrix gradient = outputLayer.backwardPropagate(record, output, hidden, new Matrix());
    	inputLayer.backwardPropagate(record, null, null, gradient);
    }
    
    /**
     * Learn parameters from data record.
     */
    @Override
    public void learn(int numIter, DataSet<DataRecord<Feature>> train, DataSet<DataRecord<Feature>> test) {
        train.shuffle();
        train.stream().forEach(x -> {
        	//Forward
        	Matrix hidden = inputLayer.forwardPropagate(x, new Matrix());
        	Matrix output = outputLayer.forwardPropagate(x, hidden);
        	
        	//Backward
        	Matrix gradient = outputLayer.backwardPropagate(x, output, hidden, new Matrix());
        	inputLayer.backwardPropagate(x, null, null, gradient);
        	
        });
    }
    
    @Override
    public double getRMSE(DataRecord<Feature> record) {
    	double target = 0.0;
    	if(numLabels == 1) {
    		target = record.getTarget();
    	}
    	else {
    		target = record.getTarget() - 1;
    	}
    	Matrix hidden = inputLayer.forwardPropagate(record, new Matrix());
    	Matrix output = outputLayer.forwardPropagate(record, hidden);
    	int label = -1;
    	double max = Double.MIN_VALUE;
    	for(int i = 0; i < numLabels; i++) {
    		if(output.get(i,0) > max) {
    			label = i;
    			max = predict(output.get(i,0));
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
}
