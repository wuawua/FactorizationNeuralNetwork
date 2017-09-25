package com.wuawua.research.fnn.utils.concurrent;

import java.util.List;
import java.util.concurrent.CountDownLatch;


public abstract class TaskManager<T> {
	private int workLength;
	private int cpuNum;
	private List<T> features;

	public TaskManager(int workLength, int cpuNum, List<T> features) {
		this.workLength = workLength;
		this.cpuNum = cpuNum;
		this.features = features;
	}

	public void start() {
		int runCpu = cpuNum < workLength ? cpuNum : 1;
		
		final CountDownLatch gate = new CountDownLatch(runCpu);
		int fregLength = (workLength + runCpu - 1) / runCpu;
		//System.out.println("fregLength: " + fregLength + " " + workLength + " " + runCpu);
		for (int cpu = 0; cpu < runCpu; cpu++) {
			final int start = cpu * fregLength;
			int tmp = (cpu + 1) * fregLength;
			final int end = tmp <= workLength ? tmp : workLength;
			Runnable task = new Runnable() {

				@Override
				public void run() {
					process(features, start, end);
					gate.countDown();
				}

			};
			ConcurenceRunner.run(task);
		}
		try {
			gate.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

	public abstract void process(List<T> features, int start, int end);
}
