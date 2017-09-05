package com.wuawua.research.fnn.math;

import org.jblas.FloatMatrix;

import com.google.common.base.Preconditions;

public class Vector {

	private int rows;
	private FloatMatrix data;
	
	public int getRows() {
		return rows;
	}
	
	public FloatMatrix getData() {
		return data;
	}

	public Vector(int size) {
		rows = size;
		data = new FloatMatrix(size, 1);
	}

	public void zero() {
		for (int i = 0; i < rows; i++) {
			data.put(i, 0.0f);
		}
	}
	
	public void fill(float value) {
		for (int i = 0; i < rows; i++) {
			data.put(i, value);
		}
	}

	public void mul(float a) {
		data.muli(a);
	}
	
	

	public void addRow(final Matrix A, int i) {
		Preconditions.checkArgument(i >= 0);
		Preconditions.checkArgument(i < A.getRows());
		Preconditions.checkArgument(rows == A.getColumns());
		for (int j = 0; j < A.getColumns(); j++) { 
			data.put(j,  data.get(j) + A.getData().get(i,j));
		}
	}

	public void addRow(final Matrix A, int i, float a) {
		Preconditions.checkArgument(i >= 0);
		Preconditions.checkArgument(i < A.getRows());
		Preconditions.checkArgument(rows == A.getColumns());
		for (int j = 0; j < A.getColumns(); j++) {
			data.put(j,  data.get(j) + a * A.getData().get(i,j));
		}
	}

	public void mul(final Matrix A, final Vector vec) {
		Preconditions.checkArgument(A.getRows() == rows);
		Preconditions.checkArgument(A.getColumns() == vec.rows);
		for (int i = 0; i < rows; i++) {
			data.put(i, 0.0f);
			for (int j = 0; j < A.getColumns(); j++) {
				data.put(i,  data.get(i) + A.getData().get(i,j) * vec.data.get(j));
			}
		}
	}
	
	public float dot(final Vector vec) {
		Preconditions.checkArgument(vec.rows == rows);
		FloatMatrix dup = data.dup();
		return dup.dot(vec.data);
	}
	
	public float norm() {
		return data.norm2();
	}
	

	public int argmax() {
		int argmax = 0;
		argmax = data.argmax();
		return argmax;
	}

	public float get(int i) {
		return data.get(i);
	}
	
	public void set(int row, float value) {
		Preconditions.checkArgument(row >= 0);
		Preconditions.checkArgument(row < rows);
		data.put(row, value);
	}
	
	public void add(int row, float value) {
		Preconditions.checkArgument(row >= 0);
		Preconditions.checkArgument(row < rows);
		data.put(row,  data.get(row) + value);
	}
	
	public String toString() {
		StringBuilder content = new StringBuilder();
		for(int ii = 0; ii < rows; ii++) {
			content.append(data.get(ii)).append(" ");
		}
		return content.toString().trim();
	}

}
