package com.wuawua.research.fnn.manager.impl;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.log4j.Logger;

import com.wuawua.research.fnn.data.Feature;
import com.wuawua.research.fnn.layer.Layer;
import com.wuawua.research.fnn.layer.impl.fm.FmInputLayer;
import com.wuawua.research.fnn.layer.impl.fm.FmOutputLayer;
import com.wuawua.research.fnn.layer.impl.fm.FmOutputLayerWithAdagrad;
import com.wuawua.research.fnn.manager.NeuralNetworkManager;
import com.wuawua.research.fnn.manager.NeuralNetworkManager.TrainThread;
import com.wuawua.research.fnn.math.Matrix;
import com.wuawua.research.fnn.math.Vector;
import com.wuawua.research.fnn.neuralnetwork.NeuralNetwork;
import com.wuawua.research.fnn.neuralnetwork.impl.FmNeuralNetwork;
import com.wuawua.research.fnn.utils.Utils;


/***
 * 
 * @author Huang Haiping
 *
 */
public class FmNeuralNetworkManager extends NeuralNetworkManager {

    private static final Logger LOG = Logger.getLogger(FmNeuralNetworkManager.class.getName());

    
    public void train(String[] args_) throws IOException {
		//Parse parameters
    	args.parseArgs(args_);
		
		labelIndex = args.labelIndex;
		
		//Read train file
		File file = new File(args.input);
		if (!(file.exists() && file.isFile() && file.canRead())) {
			throw new IOException("Input file cannot be opened! " + args.input);
		}
		
		dict.readFromFile(args.input, labelIndex, args.skips);
		
		//Read test file
		if(args.test != null && args.test.length() > 0) {
			File testFile = new File(args.input);
			if (!(testFile.exists() && testFile.isFile() && testFile.canRead())) {
				throw new IOException("Input test file cannot be opened! " + args.test);
			}
			dict.readFromFile(args.test, labelIndex, args.skips);
		}
		
		//Classification task has a lot of labels,  example 1,2,3,4,5
		//ranking task has only one label, ranking task is Logistic regression. 
		numLabels = dict.getNumLabels();
		if(args.numLabels > 0) {
			numLabels = args.numLabels;
		}
		
		Random rnd = new Random();
		float sdev = 0.1f; 
		
		System.out.println(dict.getWords());
		System.out.println(args.lr + " " + args.reg + " " + args.numLabels);
		
		//Initialize InputLayer parameters
		inputWeights = new Matrix(dict.getWords() * numLabels, args.dim);
		inputBias = new Vector(dict.getWords());
		inputReg = new Vector(dict.getWords());
		for (int ii = 0; ii < inputWeights.getRows(); ii++) {
            for (int jj = 0; jj < inputWeights.getColumns(); jj++) {
            	inputWeights.set(ii, jj, (float)rnd.nextGaussian() * sdev);
            }
        }
		inputBias.zero();
		inputReg.fill(args.reg);
		
		//Initialize OutputLayer parameters
		outputWeights = new Matrix(dict.getWords() * numLabels, args.dim);
		outputBias = new Vector(dict.getWords());
		outputReg = new Vector(dict.getWords());
		for (int ii = 0; ii < outputWeights.getRows(); ii++) {
            for (int jj = 0; jj < outputWeights.getColumns(); jj++) {
            	outputWeights.set(ii, jj, (float)rnd.nextGaussian() * sdev);
            }
        }
		outputBias.zero();
		outputReg.fill(args.reg);

		info.start = System.currentTimeMillis();
		long t0 = System.currentTimeMillis();

		threadCount = args.thread;
		tokenCount = new AtomicLong(0l);

		
		
		long fileSize = Utils.sizeLine(args.input);
		for (int i = 0; i < args.thread; i++) {
			
			Layer<Feature> inputLayer = new FmInputLayer( args.dim, args.lr, inputBias, inputReg, inputWeights,inputReg);
			Layer<Feature> outputLayer = new FmOutputLayer(args.dim, numLabels, args.lr, outputBias, inputReg, outputWeights, inputReg);
		    NeuralNetwork<Feature> nn = new FmNeuralNetwork(numLabels, inputLayer, outputLayer);
		    
		    
			new TrainThread(this, args, dict, inputWeights, inputBias, 
					inputReg,  outputWeights, outputBias, outputReg, i, fileSize,
					inputLayer, outputLayer, nn).start();
		}

		synchronized (this) {
			while (threadCount > 0) {
				try {
					wait();
				} catch (InterruptedException ignored) {
				}
			}
		}
		
		
		//runTrain();
		
		long trainTime = (System.currentTimeMillis() - t0) / 1000;
		System.out.printf("Train time: %d sec\n", trainTime);

		if (!Utils.isEmpty(args.output)) {
			//saveModel(dict, input, output);
			//saveVectors(dict, input, output);
			saveVectors(dict);
		}
	}
}
