package com.wuawua.research.fnn;

import com.wuawua.research.fnn.manager.impl.FfmNeuralNetworkManager;
import com.wuawua.research.fnn.manager.impl.FmNeuralNetworkManager;

public class FfmMain {

	public void printUsage() {
		System.out.print(
				"usage: java -jar fnn.jar <command> <args>\n\n" 
				+ "The commands supported by fasttext are:\n\n"
				+ " supervised    train a supervised classifier\n" 
				+ " test          evaluate a supervised classifier\n"
				+ " predict       predict most likely label\n"
				+ " predict-prob  predict most likely label with probabilities\n" 
				+ " skipgram      train a skipgram model\n"
				+ " cbow          train a cbow model\n" 
				+ " print-vectors print vectors given a trained model\n");
	}

	public void printTestUsage() {
		System.out.print("usage: java -jar fnn.jar test <model> <test-data>\n\n" 
						+ " <model> model filename\n"
						+ " <test-data> test data filename\n");
	}

	public void printPredictUsage() {
		System.out.print("usage: java -jar fnn.jar predict <model> <test-data>\n\n" 
	                     + " <model> model filename\n"
				         + " <test-data> test data filename\n");
	}

	public void printPrintVectorsUsage() {
		System.out.print("usage: java -jar fnn.jar print-vectors <model>\n\n" 
						+ " <model> model filename\n");
	}
	
	public void run(String[] args) {
		FfmNeuralNetworkManager network = new FfmNeuralNetworkManager();
		if (args.length < 2) {
			printUsage();
			System.exit(1);
		}

		try {
			String command = args[0];
			if ("supervised".equalsIgnoreCase(command)) {
				network.train(args);
			} else {
				printUsage();
				System.exit(1);
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		System.exit(0);
	}
	
	public static void main(String[] args) {
		FfmMain fnn = new FfmMain();
		
		args = new String[] {
			"supervised",
			"-input", "data/u1.feature.base.txt",
			"-test", "data/u1.feature.test.txt",
			"-output", "data/u1.model",
			"-dim", "150", 
			"-lr", "0.01", 
			"-minCount", "0",
			"-epoch", "200",
			"-thread", "1",
			"-labelIndex", "0",
			"-numLabels", "1",
			"-reg", "0.1",
			"-skips", "4,5,6,8,9",
		};
				
		fnn.run(args);
	}

}
