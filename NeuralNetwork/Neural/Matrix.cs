using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Neural
{
    class Matrix
    {
        public int _rows;
        public int _cols;
        public double[,] _data;

        public Matrix(int rows, int cols)
        {
            _rows = rows;
            _cols = cols;

            _data = new double[rows, cols];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    _data[i, j] = 0;
                }
            }
        }

        public void Randomize()
        {
            for (var i = 0; i < _rows; i++)
            {
                for (var j = 0; j < _cols; j++)
                {
                    _data[i, j] = (double)new Random().NextDouble() * 2 - 1;
                }
            }
        }

        public void Add(Matrix v)
        {
            for (var i = 0; i < this._rows; i++)
            {
                for (var j = 0; j < this._cols; j++)
                {
                    this._data[i, j] += v._data[i, j];
                }
            }
        }

        public double[] ToArray()
        {
            List<double> arr = new List<double>();

            for (var i = 0; i < _rows; i++)
            {
                for (var j = 0; j < _cols; j++)
                {
                    arr.Add(_data[i, j]);
                }
            }

            return arr.ToArray();
        }

        public void ApplyFunc(Func<double, double> f)
        {
            for (var i = 0; i < _rows; i++)
            {
                for (var j = 0; j < _cols; j++)
                {
                    double val = _data[i, j];
                    _data[i, j] = f(val);
                }
            }
        }

        public static Matrix ApplyFunc(Matrix m, Func<double, double> f)
        {
            var result = new Matrix(m._rows, m._cols);

            for (var i = 0; i < m._rows; i++)
            {
                for (var j = 0; j < m._cols; j++)
                {
                    double val = m._data[i, j];
                    result._data[i, j] = f(val);
                }
            }

            return result;
        }

        public static Matrix Transpose(Matrix m)
        {
            var result = new Matrix(m._cols, m._rows);
            for (var i = 0; i < m._rows; i++)
            {
                for (var j = 0; j < m._cols; j++)
                {
                    result._data[j, i] = m._data[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Retorna o produto da multiplicação de matrizes.
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        public static Matrix Multiply(Matrix a, Matrix b)
        {
            var result = new Matrix(a._rows, b._cols);

            for (var i = 0; i < result._rows; i++)
                for (var j = 0; j < result._cols; j++)
                {
                    double sum = 0;
                    for (var k = 0; k < a._cols; k++)
                    {
                        sum += a._data[i, k] * b._data[k, j];
                    }

                    result._data[i, j] = sum;
                }

            return result;
        }

        public static Matrix MultiplyScalar(Matrix a, double v)
        {
            var result = new Matrix(a._rows, a._cols);

            for (var i = 0; i < result._rows; i++)
                for (var j = 0; j < result._cols; j++)
                {
                    result._data[i, j] = a._data[i, j] * v;
                }

            return result;
        }

        public static Matrix Subtract(Matrix a, Matrix b)
        {
            var result = new Matrix(a._rows, a._cols);

            for (var i = 0; i < result._rows; i++)
                for (var j = 0; j < result._cols; j++)
                {
                    result._data[i, j] = a._data[i, j] - b._data[i, j];
                }

            return result;
        }

        public static Matrix FromArray(double[] arr)
        {
            var mat = new Matrix(arr.Length, 1);

            for (var i = 0; i < arr.Length; i++)
            {
                mat._data[i, 0] = arr[i];
            }

            return mat;
        }

    }
}
