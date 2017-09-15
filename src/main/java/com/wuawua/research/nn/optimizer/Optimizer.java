package com.wuawua.research.nn.optimizer;

import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;

/***
 * Optimizer of backward propagate
 * @author Huang Haiping
 *
 * @param <T>
 */
public interface Optimizer<T> {
	
	public void register(Layer<T> layer);

	public void update(Vector gradient, int index, float xi);
	
	public void update(Matrix gradient, int index, float xi, int gradienIndex);

	public Optimizer<T> dup();
}
