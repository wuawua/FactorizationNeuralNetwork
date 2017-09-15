package com.wuawua.research.fnn.manager.impl;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.log4j.Logger;

import com.wuawua.research.fnn.manager.NeuralNetworkManager;
import com.wuawua.research.fnn.manager.NeuralNetworkManager.TrainThread;
import com.wuawua.research.fnn.utils.Utils;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.layer.impl.ffm.FfmInputLayer;
import com.wuawua.research.nn.layer.impl.ffm.FfmOutputLayer;
import com.wuawua.research.nn.layer.impl.fm.FmInputLayer;
import com.wuawua.research.nn.layer.impl.fm.FmOutputLayer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.neuralnetwork.NeuralNetwork;
import com.wuawua.research.nn.neuralnetwork.impl.FfmNeuralNetwork;
import com.wuawua.research.nn.neuralnetwork.impl.FmNeuralNetwork;
import com.wuawua.research.nn.optimizer.Adagrad;
import com.wuawua.research.nn.optimizer.Optimizer;
import com.wuawua.research.nn.optimizer.SGD;


/***
 * 
 * @author Huang Haiping
 *
 */
public class FfmNeuralNetworkManager extends NeuralNetworkManager {

    private static final Logger LOG = Logger.getLogger(FfmNeuralNetworkManager.class.getName());

    
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
		
		int numFields = dict.getNumFields();
		System.out.println("numFields: " + numFields);
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
		inputWeights = new Matrix(dict.getWords() * numLabels * numFields, args.dim);
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
		outputWeights = new Matrix(dict.getWords() * numLabels * numFields, args.dim);
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
			
			Optimizer<Feature> optimizer = new SGD<Feature>(0, 0, args.lr);
			Layer<Feature> inputLayer = new FfmInputLayer( args.dim, numFields, args.lr, inputBias, inputReg, inputWeights,inputReg, optimizer);
			Layer<Feature> outputLayer = new FfmOutputLayer(args.dim, numLabels, numFields, args.lr, outputBias, inputReg, outputWeights, inputReg);
		    NeuralNetwork<Feature> nn = new FfmNeuralNetwork(numLabels, inputLayer, outputLayer);
		    
		    
			new TrainThread(this, args, dict, i, fileSize, inputLayer, outputLayer, nn).start();
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
