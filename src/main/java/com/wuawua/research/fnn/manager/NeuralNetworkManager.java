package com.wuawua.research.fnn.manager;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.log4j.Logger;

import com.wuawua.research.fnn.Args;
import com.wuawua.research.fnn.model.Dictionary;
import com.wuawua.research.fnn.utils.Utils;
import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.DataSet;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.data.SimpleDataSet;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.neuralnetwork.NeuralNetwork;



/***
 * 
 * @author Huang Haiping
 *
 */
public class NeuralNetworkManager {

	protected static final Logger LOG = Logger.getLogger(NeuralNetworkManager.class.getName());

	protected int numLabels;
	protected int numFields;
    int trainNum = 0;
    
    protected static Logger logger = Logger.getLogger(NeuralNetworkManager.class);

    protected Args args = new Args();
    protected Dictionary dict = new Dictionary(args);
    protected Matrix inputWeights = new Matrix();
    protected Vector inputBias;
    protected Vector inputReg;
	
    protected Matrix outputWeights = new Matrix();
    protected Vector outputBias;
    protected Vector outputReg;

    protected int labelIndex = 0;
    double rmse = 0.0;
	int counter = 0;
	//List<Integer> skips;
	
    protected final int SINGLE_LABEL_NUMBER = 1;

	
    protected int threadCount;
    protected AtomicLong tokenCount = new AtomicLong(0l);

	public class Info {
		public long start = 0;
		public AtomicLong allWords = new AtomicLong(0l);
		public AtomicLong allN = new AtomicLong(0l);
		public double allLoss = 0.0;
	}

	public class MutableDouble {
		private double value;

		public MutableDouble(double value) {
			this.value = value;
		}

		public void set(double value) {
			this.value = value;
		}

		public double doubleValue() {
			return value;
		}

		public void incrementDouble(double value) {
			this.value += value;
		}
	}

	public Info info = new Info();
	
    
    public void train(String[] args_) throws IOException {
	}

	public class TrainThread extends Thread {
		final NeuralNetworkManager network;
		Dictionary dict;
		Matrix inputWeights;
		Vector inputBias;
		Vector inputReg;
		Matrix outputWeights;
		Vector outputBias;
		Vector outputReg;
		int threadId;
		long fileSize;
		Layer<Feature> inputLayer;
		Layer<Feature> outputLayer;
		NeuralNetwork<Feature> nn;

		public TrainThread(NeuralNetworkManager network, Args args, Dictionary dict, 
				int threadId, long fileSize, Layer<Feature> inputLayer, Layer<Feature> outputLayer, 
				NeuralNetwork<Feature> nn) {
			this.network = network;
			this.dict = dict;
			this.threadId = threadId;
			this.fileSize = fileSize;
			this.inputLayer = inputLayer;
			this.outputLayer = outputLayer;
			this.nn = nn;
		}

		public void run() {
			if (logger.isDebugEnabled()) {
				logger.debug("thread: " + threadId + " RUNNING!");
			}
			
			long begin = threadId * fileSize / args.thread;
			long end = (threadId + 1) * fileSize / args.thread;
			
			SimpleDataSet<DataRecord<Feature>> trainSet;
			try {
				trainSet = getDataSet(args, dict, args.input, begin, end, labelIndex);
				
				SimpleDataSet<DataRecord<Feature>> testSet = null;
				if(args.test != null) {
					testSet = getDataSet(args, dict, args.test, begin, end, labelIndex);
				}
				
				for(int epoch = 1; epoch <= args.epoch; epoch++) {				
					
					//Train neural network
					nn.learn(epoch, trainSet, testSet);
					
					
					//Display neural network model performance.
					if (threadId == 0 && epoch % 10 == 0) {
						//System.out.println("Epoch finish: " + epoch);
						try {
							double trainRMSE = test(args.input, nn);
							double testRMSE = test(args.test, nn);
							System.out.printf("epoch %d, train RMSE %.8f, test RMSE %.8f%n", epoch, trainRMSE, testRMSE);
						} catch (IOException e) {;
						}	
					}
				} //End of for
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			
			synchronized (network) {
				if (logger.isDebugEnabled()) {
					logger.debug("thread: " + threadId + " EXIT!");
				}
				network.threadCount--;
				network.notify();
			}
		}
	}
	
	
	
	public void runTrain(long fileSize, NeuralNetwork<Feature> nn) {
		//if (logger.isDebugEnabled()) {
		//	logger.debug("thread: " + threadId + " RUNNING!");
		//}
		
		long begin = 0; //threadId * fileSize / args.thread;
		long end = fileSize; //(threadId + 1) * fileSize / args.thread;
		
		
		SimpleDataSet<DataRecord<Feature>> trainSet;
		try {
			trainSet = getDataSet(args, dict, args.input, begin, end, labelIndex);
			
			SimpleDataSet<DataRecord<Feature>> testSet = null;
			if(args.test != null) {
				testSet = getDataSet(args, dict, args.test, begin, end, labelIndex);
			}
			
			for(int epoch = 1; epoch <= args.epoch; epoch++) {				
				
				//Train neural network
				nn.learn(epoch, trainSet, testSet);
				
				
				//Display neural network model performance.
				//if (threadId == 0 && epoch % 10 == 0) {
					try {
						double trainRMSE = test(args.input, nn);
						double testRMSE = test(args.test, nn);
						System.out.printf("epoch %d, train RMSE %.8f, test RMSE %.8f%n", epoch, trainRMSE, testRMSE);
					} catch (IOException e) {;
					}	
				//}
			} //End of for
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		//synchronized (network) {
		//	if (logger.isDebugEnabled()) {
		//		logger.debug("thread: " + threadId + " EXIT!");
		//	}
		//	network.threadCount--;
		//	network.notify();
		//}
	}
	/**
	 * Read data set from file
	 * @param args
	 * @param dict
	 * @param file
	 * @param begin
	 * @param end
	 * @param labelIndex
	 * @return
	 * @throws IOException
	 */
	public SimpleDataSet<DataRecord<Feature>> getDataSet(Args args, Dictionary dict, 
			String file, long begin, long end, int labelIndex) throws IOException {
    	SimpleDataSet<DataRecord<Feature>> dataset = new SimpleDataSet<DataRecord<Feature>>(dict.getWords());
    	BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(args.input));
			Utils.seek(br, begin);
			String lineString;
			
			while ( begin < end) {						
				//Read a line of data record.
				lineString = br.readLine();
				if (lineString == null) {
					try {
						br.close();
						br = new BufferedReader(new FileReader(args.input));
						if (logger.isDebugEnabled()) {
							logger.debug("Input file reloaded!");
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
					lineString = br.readLine();
				}
				while (Utils.isEmpty(lineString) || lineString.startsWith("#")) {
					lineString = br.readLine();
				}
				
				//Data record
				DataRecord<Feature> record = dict.getDataRecord(lineString, labelIndex, args.skips);
				normalize(record, true);
				dataset.add(record);
            	begin++;
			}
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		} finally {
			if (br != null)
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		} //End of try


        return dataset;
    }
	
	public void normalize(DataRecord<Feature> record, boolean normal) {
		if(normal) {
			float norm = 0;
			for(Feature feature : record.getFeatures()) {
				norm += feature.getValue() * feature.getValue();
			}
			norm = norm/((float)record.getFeatureSize());
			for(Feature feature : record.getFeatures()) {
				float value = feature.getValue() / (float) Math.sqrt(norm);
				feature.setValue(value);
			}
		} else {
		}		
	}
	
	
	public double test(String filename, NeuralNetwork<Feature> nn) throws IOException {
		double rmse = 0.0;
		int counter = 0;
		File file = new File(filename);
		if (!(file.exists() && file.isFile() && file.canRead())) {
			throw new IOException("Test file cannot be opened!");
		}
		
		FileInputStream fis = new FileInputStream(file);
		BufferedReader dis = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
		try {
			String lineString;
			while ((lineString = dis.readLine()) != null) {
				DataRecord<Feature> record = dict.getDataRecord(lineString, args.labelIndex, args.skips);
				normalize(record, true);
				rmse += nn.getRMSE(record);
        		counter++;
			}
		} finally {
			dis.close();
			fis.close();
		}
		
		rmse = Math.sqrt(rmse/counter); 
		return rmse;
    }
	
	public double test(DataSet<DataRecord<Feature>> dataset, NeuralNetwork<Feature> nn) throws IOException {
		rmse = 0.0;
		counter = 0;
		
		dataset.stream().forEach(x -> {
			//DataRecord<Feature> record = dict.getDataRecord(lineString, args.labelIndex, args.skips);
			//normalize(record, true);
			rmse += nn.getRMSE(x);
    		counter++;
		});
			
		
		rmse = Math.sqrt(rmse/counter); 
		return rmse;
    }
	
	
    
    public void saveModel(Dictionary dict, Matrix input, Matrix output) throws IOException {
		File file = new File(args.output + ".bin");
		if (file.exists()) {
			file.delete();
		}
		logger.info("Saving model to " + file.getAbsolutePath().toString());
		FileOutputStream fos = new FileOutputStream(file);
		OutputStream ofs = new DataOutputStream(fos);
		try {
			logger.debug("writing args");
			args.save(ofs);
			logger.debug("writing dict");
			dict.save(ofs);
			logger.debug("writing input");
			input.save(ofs);
			logger.debug("writing output");
			output.save(ofs);
		} finally {
			ofs.flush();
			ofs.close();
		}
	}
	
	public void loadModel(String filename) throws IOException {
		DataInputStream dis = null;
		BufferedInputStream bis = null;
		
		try {
			File file = new File(filename);
			if (!(file.exists() && file.isFile() && file.canRead())) {
				throw new IOException("Model file cannot be opened for loading!");
			}
			bis = new BufferedInputStream(new FileInputStream(file));
			dis = new DataInputStream(bis);
			
			loadModel(dis);
		} catch (FileNotFoundException e) {
			logger.debug(e.getLocalizedMessage());
		} catch (IOException e) {
			logger.debug(e.getLocalizedMessage());
		} finally {
			bis.close();
			dis.close();
		}
		

	}
	
	public void loadModel(DataInputStream dis) throws IOException {
		args = new Args();
		dict = new Dictionary(args);
		inputWeights = new Matrix();
		outputWeights = new Matrix();
		
		args.load(dis);
		dict.load(dis);
		inputWeights.load(dis);
		outputWeights.load(dis);
		
		logger.info("loadModel done!");

	}
	
	public void printInfo(float progress, float loss_) {
		float loss = (float) (info.allLoss / info.allN.get());
		float t = (float) ((System.currentTimeMillis() - info.start) / 1000);
		float wst = (float) (info.allWords.get() / t);
		double lr = args.lr * ( 1.0 - progress );
		int eta = (int) (t / progress * (1 - progress) / args.thread);
		int etah = eta / 3600;
		int etam = (eta - etah * 3600) / 60;
		System.out.printf("\rProgress: %.1f%% words/sec/thread: %d lr: %.6f loss: %.6f eta: %d h %d m", 100 * progress,
				(int) wst, lr, loss, etah, etam);
	}
	

	public void saveVectors(Dictionary dict) throws IOException {
		File file = new File(args.output + ".vec");
		if (file.exists()) {
			file.delete();
		}
		logger.info("Saving Vectors to " + file.getAbsolutePath().toString());
		Writer writer = new FileWriter(file);
		try {
			writer.write(dict.getWords());
			writer.write(" ");
			writer.write(args.dim);
			writer.write("\n");
			//Vector vec = new Vector(args.dim);
			DecimalFormat df = new DecimalFormat("#####.#");
			for (int i = 0; i < dict.getWords(); i++) {
				String word = dict.getWord(i);
				int id = dict.getId(word);
				//getVector(dict, input, vec, word);
				writer.write(word);
				writer.write(" ");
				writer.write(df.format(id));
				//for (int j = 0; i < vec.m_; i++) {
				//	writer.write(df.format(vec.data_.get(j)));
				//	writer.write(" ");
				//}
				writer.write("\n");
			}
		} finally {
			writer.flush();
			writer.close();
		}
	}
}
