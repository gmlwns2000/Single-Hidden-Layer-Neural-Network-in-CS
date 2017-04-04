using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra.Mkl;
using System;
using System.Text;

namespace NNTest1Console
{
    public delegate float ActivationFunction(float X);

    public class SimpleNeuralNet
    {
        float _learningRate;
        int _inputCount;
        int _hiddenCount;
        int _outputCount;

        Matrix<float> wih;
        Matrix<float> who;

        ActivationFunction _activationFunction;
        Random rand;

        public static float Sigmoid(float X)
        {
            return (float)(1 / (1 + Math.Pow(Math.E, -X)));
        }

        public SimpleNeuralNet(int inputCount, int hiddenCount, int outputCount, float rate)
        {
            Control.UseNativeMKL();
            string version = Control.LinearAlgebraProvider.ToString();

            _learningRate = rate;

            _inputCount = inputCount;
            _outputCount = outputCount;
            _hiddenCount = hiddenCount;

            rand = new Random((int)DateTime.UtcNow.TimeOfDay.TotalMilliseconds);

            wih = Matrix<float>.Build.DenseOfColumnMajor(_hiddenCount, _inputCount, new float[_hiddenCount * _inputCount]);
            wih = MatrixRand(wih);

            who = Matrix<float>.Build.DenseOfColumnMajor(_outputCount, _hiddenCount, new float[_outputCount * _hiddenCount]);
            who = MatrixRand(who);

            _activationFunction = Sigmoid;
        }

        Matrix<float> hidden_inputs;
        Matrix<float> hidden_outputs;
        Matrix<float> final_inputs;
        Matrix<float> final_outputs;
        Matrix<float> output_error;
        Matrix<float> hidden_error;
        Matrix<float> who_adjust;
        Matrix<float> wih_adjust;
        public void Train(float[] inputArray, float[] targetArray)
        {
            Matrix<float> inputs = CreateMatrix.DenseOfColumnMajor(inputArray.Length, 1, inputArray);

            Matrix<float> target = CreateMatrix.DenseOfColumnMajor(targetArray.Length, 1, targetArray);

            if(hidden_inputs == null)
            {
                hidden_inputs = CreateMatrix.DenseOfColumnMajor(_hiddenCount, 1, new float[_hiddenCount]);
            }
            wih.Multiply(inputs, hidden_inputs);

            if(hidden_outputs == null)
            {
                hidden_outputs = CreateMatrix.DenseOfColumnMajor(_hiddenCount, 1, new float[_hiddenCount]);
            }
            hidden_outputs = Activate(hidden_inputs);

            if(final_inputs == null)
            {
                final_inputs = CreateMatrix.DenseOfColumnMajor(_outputCount, 1, new float[_outputCount]);
            }
            who.Multiply(hidden_outputs, final_inputs);

            if (final_outputs == null)
            {
                final_outputs = CreateMatrix.DenseOfColumnMajor(_outputCount, 1, new float[_outputCount]);
            }
            final_outputs = Activate(final_inputs);

            if(output_error == null)
            {
                output_error = CreateMatrix.DenseOfColumnMajor(_outputCount, 1, new float[_outputCount]);
            }
            target.Subtract(final_outputs, output_error);

            if(hidden_error == null)
            {
                hidden_error = CreateMatrix.DenseOfColumnMajor(_hiddenCount, 1, new float[_hiddenCount]);
            }
            who.TransposeThisAndMultiply(output_error, hidden_error);

            if (who_adjust == null)
            {
                who_adjust = CreateMatrix.DenseOfColumnMajor(_outputCount, _hiddenCount, new float[_outputCount * _hiddenCount]);
            }
            MatrixMul(MatrixMul(output_error, final_outputs), (1.0f - final_outputs)).TransposeAndMultiply(hidden_outputs, who_adjust);
            who_adjust.Multiply(_learningRate, who_adjust);
            who.Add(who_adjust, who);

            if (wih_adjust == null)
            {
                wih_adjust = CreateMatrix.DenseOfColumnMajor(_hiddenCount, _inputCount, new float[_hiddenCount * _inputCount]);
            }
            MatrixMul(MatrixMul(hidden_error, hidden_outputs), (1.0f - hidden_outputs)).TransposeAndMultiply(inputs, wih_adjust);
            wih_adjust.Multiply(_learningRate, wih_adjust);
            wih.Add(wih_adjust, wih);
        }

        public int Query(float[] inputArray)
        {
            Matrix<float> inputs = CreateMatrix.DenseOfColumnMajor(784, 1, inputArray);

            var hidden_inputs = wih * inputs;
            var hidden_outputs = Activate(hidden_inputs);

            var final_inputs = who * hidden_outputs;
            var final_outputs = Activate(final_inputs);

            float[] fArray = final_outputs.ToRowMajorArray();

            int maxInd = -1;
            float max = 0;

            for (int i = 0; i < fArray.Length; i++)
            {
                if(fArray[i] > max)
                {
                    max = fArray[i];
                    maxInd = i;
                }
            }

            Printer.WriteLine(string.Format("Outputs: {0}, Answer: {1}", ArrayToChar(final_outputs.ToColumnMajorArray()), maxInd));

            return maxInd;
        }

        private Matrix<float> Activate(Matrix<float> mat)
        {
            for (int row = 0; row < mat.RowCount; row++)
            {
                for (int i = 0; i < mat.ColumnCount; i++)
                {
                    mat[row, i] = _activationFunction(mat[row, i]);
                }
            }

            return mat;
        }

        private Matrix<float> MatrixMul(Matrix<float> copied, Matrix<float> RightMat)
        {
            int copiedRowCount = copied.RowCount;
            int copiedColumnCount = copied.ColumnCount;

            for (int row = 0; row < copiedRowCount; row++)
            {
                for (int column = 0; column < copiedColumnCount; column++)
                {
                    copied[row, column] *= RightMat[row, column];
                }
            }

            return copied;
        }

        private Matrix<float> MatrixRand(Matrix<float> Mat)
        {
            for (int row = 0; row < Mat.RowCount; row++)
            {
                for (int column = 0; column < Mat.ColumnCount; column++)
                {
                    Mat[row, column] = (float)rand.Next() / int.MaxValue - 0.5f;
                }
            }

            return Mat;
        }

        StringBuilder b = new StringBuilder();
        private string ArrayToChar(float[] arr)
        {
            b.Clear();

            b.Append("{");

            foreach(float f in arr)
            {
                b.Append(" \"");
                b.Append(f.ToString("0.0000"));
                b.Append("\"");
            }

            b.Append(" }");

            return b.ToString();
        }
    }
}
