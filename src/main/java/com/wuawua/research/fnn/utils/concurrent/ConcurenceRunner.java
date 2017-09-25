package com.wuawua.research.fnn.utils.concurrent;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ConcurenceRunner {

	private static final ExecutorService exec;
	public static final int cpuNum;
	static {
		cpuNum = Runtime.getRuntime().availableProcessors();
		exec = Executors.newFixedThreadPool(cpuNum);
	}

	public static void run(Runnable task) {
		exec.execute(task);
	}

	public static void stop() {
		exec.shutdown();
	}
	
	public static int getCpuNum() {
		return cpuNum;
	}
	
}
