package com.wuawua.research.fnn.math;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.Well19937c;
import org.apache.log4j.Logger;
import org.jblas.FloatMatrix;

import com.google.common.base.Preconditions;
import com.wuawua.research.fnn.utils.IOUtil;


public class Matrix {

	private static Logger logger = Logger.getLogger(Matrix.class);

	private int rows = 0; 
	private int columns = 0; 
	private FloatMatrix data;

	public Matrix() {
	}
	
	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		this.data = new FloatMatrix(rows, columns);
	}

	public Matrix(final Matrix other) {
		this.rows = other.rows;
		this.columns = other.columns;
		this.data = new FloatMatrix(rows, columns);
		data.copy(other.data);
	}
	
	public int getRows() {
		return rows;
	}
	
	public int getColumns() {
		return columns;
	}
	
	public FloatMatrix getData() {
		return data;
	}

	public void zero() {		
		for (int i = 0; i < rows * columns; i++) {
			data.put(i, 0.0f);
        }
	}

	public void uniform(float a) {
		UniformRealDistribution urd = new UniformRealDistribution(new Well19937c(1), -a, a);
		for (int i = 0; i < rows * columns; i++) {
			data.put(i, (float)urd.sample());
        }
	}
	
	public float get(int row, int column) {
		Preconditions.checkArgument(row >= 0);
		Preconditions.checkArgument(row < rows);
		Preconditions.checkArgument(column >= 0);
		Preconditions.checkArgument(column < columns);
		return data.get(row, column);
	}
	
	public void add(int row, int column, float value) {
		Preconditions.checkArgument(row >= 0);
		Preconditions.checkArgument(row < rows);
		Preconditions.checkArgument(column >= 0);
		Preconditions.checkArgument(column < columns);
		data.put(row, column, data.get(row, column) + value);
	}
	
	public void set(int row, int column, float value) {
		Preconditions.checkArgument(row >= 0);
		Preconditions.checkArgument(row < rows);
		Preconditions.checkArgument(column >= 0);
		Preconditions.checkArgument(column < columns);
		data.put(row, column, value);
	}

	public void addRow(final Vector vec, int i, float a) {
		Preconditions.checkArgument(i >= 0);
		Preconditions.checkArgument(i < rows);
		Preconditions.checkArgument(vec.getRows() == columns);
		for (int j = 0; j < columns; j++) {
			data.put(i, j, data.get(i,j) + a * vec.getData().get(j));
		}
	}
	
	public float dotRow(final Vector vec, int i) {
		Preconditions.checkArgument(i >= 0);
		Preconditions.checkArgument(i < rows);
		Preconditions.checkArgument(vec.getRows() == columns);
		float d = 0.0f;
		for (int j = 0; j < columns; j++) {
			d += data.get(i,j) * vec.getData().get(j);
		}
		return d;
	}
	
	public float dotRow(final FloatMatrix vec, int i) {
		Preconditions.checkArgument(i >= 0);
		Preconditions.checkArgument(i < rows);
		Preconditions.checkArgument(vec.rows == columns);
		float d = 0.0f;
		for (int j = 0; j < columns; j++) {
			d += data.get(i,j) * vec.get(j);
		}
		return d;
	}
	

	public void load(InputStream input) throws IOException {
		rows = (int) IOUtil.readLong(input);
		columns = (int) IOUtil.readLong(input);
		data = new FloatMatrix(rows,columns);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data.put(i, j, IOUtil.readFloat(input));
			}
		}
		
		if (logger.isDebugEnabled()) {
			logger.debug("Matrix loal m_: " + rows);
			logger.debug("Matrix loal n_: " + columns);
			StringBuilder strBuilder = new StringBuilder("line1:");
			for (int j = 0; j < columns; j++) {
				strBuilder.append(" ").append(data.get(0,j));
			}
			logger.debug(strBuilder.toString());
		}
	}

	public void save(OutputStream ofs) throws IOException {
		ofs.write(IOUtil.longToByteArray(rows));
		ofs.write(IOUtil.longToByteArray(columns));
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				ofs.write(IOUtil.floatToByteArray(data.get(i,j)));
			}
		}
	}

}
